/*
 * Task 2 - POSIX Threads: Finding Prime Numbers
 * FIT3143 Lab 1 (Week 4)   [REVISED for Lab 2 / Week 8]
 *
 * Parallel version of Task 1 using POSIX Threads (pthreads).
 * Outputs to the console if n < 100, or to output.txt if n >= 100.
 *
 * How it works:
 * - Threads take turns processing fixed-size chunks of numbers in a round-robin
 *   style (e.g., Thread 0 takes chunk 0, Thread 1 takes chunk 1, and so on).
 * - Since larger numbers take more math to check, simply splitting the range
 *   into equal halves would leave the thread with the biggest numbers doing way
 *   more work. The round-robin approach gives every thread a fair mix of small
 *   and large numbers, balancing the load without needing slow mutex locks.
 * - Threads write their results straight into a shared array (flags). Since each
 *   thread only writes to its assigned indices, they don't step on each other,
 *   which avoids race conditions.
 * - The flags are then compacted into a sorted list, IN PARALLEL (see below).
 *
 * ---- Revisions for Lab 2 (see CHANGES.md) ----------------------------------
 * 1. The serial O(n) flag scan is gone. Marker feedback: "computation is
 *    followed by a serial flag scan and array construction". Compaction is now
 *    a three-phase parallel operation using a prefix sum over per-thread counts:
 *
 *      Phase A  each thread counts set flags in its own CONTIGUOUS block
 *               [klo, khi) of the flag array
 *      Phase B  one thread computes the exclusive prefix sum of those counts,
 *               giving every thread the exact offset at which its primes
 *               begin in the output array (O(p) work, p = thread count)
 *      Phase C  each thread copies its own block's primes to primes[offset...]
 *
 *    Sorted order is preserved for free: blocks are contiguous and ascending,
 *    and each thread walks its own block in ascending order, so thread i's
 *    output is entirely below thread i+1's. No sort is needed afterwards.
 *
 *    Note the two different partitionings. The SEARCH uses cyclic chunks,
 *    because per-candidate cost grows with k and cyclic assignment mixes cheap
 *    and expensive candidates. The COMPACTION uses contiguous blocks, because
 *    its cost is uniform per element and contiguity is what makes the output
 *    sorted. Different phases, different best partitioning.
 *
 * 2. Timing: the timed region now runs from just before thread creation to
 *    the point where the sorted prime list exists in memory. Marker feedback:
 *    "timing includes thread creation/join overhead but not result
 *    construction". Both are now included, matching task1.c exactly, so the
 *    speedup figures are honest.
 *
 * 3. Threads are created once and reused across all three phases via a
 *    pthread_barrier_t, rather than being created and joined per phase.
 *
 * 4. Output buffer sized by the Rosser-Schoenfeld bound instead of a
 *    count-then-allocate pass (that counting pass was itself serial O(n)).
 *
 * Compile: gcc task2.c -o task2 -O2 -pthread -lm
 * Run:     ./task2 <n> <number_of_threads>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

/* Odd candidates per chunk. Matches Task 3's schedule(dynamic, 1000) so the
 * two parallel schemes are directly comparable. */
#define CHUNK 1000

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec / 1e9;
}

/* Rosser-Schoenfeld bound: pi(x) < 1.25506 * x / ln(x) for x > 1. */
static long prime_count_bound(int n) {
    if (n < 100) {
        return n;
    }
    return (long) (1.25506 * (double) n / log((double) n)) + 1;
}

/* Returns 1 if k is prime, 0 otherwise. Only checks divisors up to sqrt(k),
 * and only odd divisors, since k is guaranteed odd when this is called. */
int is_prime(int k) {
    if (k < 2) {
        return 0;
    }
    if (k == 2) {
        return 1;
    }
    if (k % 2 == 0) {
        return 0;
    }

    int limit = (int) sqrt((double) k);
    for (int divisor = 3; divisor <= limit; divisor += 2) {
        if (k % divisor == 0) {
            return 0;
        }
    }
    return 1;
}

/* Per-thread state. flags, primes, offsets and barrier are shared; each
 * thread only ever writes to indices it owns, so no mutex is required.
 * search_time records time spent in the search phase only, used afterwards
 * to report load imbalance. compact_time does the same for compaction. */
typedef struct {
    int id;
    int num_threads;
    int n;
    char *flags;
    int *primes;
    int *offsets;
    long num_candidates;
    pthread_barrier_t *barrier;
    double search_time;
    double compact_time;
} ThreadData;

void *find_primes(void *arg) {
    ThreadData *d = (ThreadData *) arg;

    /* ---- Phase 1: prime search, cyclic chunk partitioning ---------------
     * Chunk c belongs to this thread iff c % num_threads == id. */
    double t0 = now_seconds();

    for (long c = d->id; c * CHUNK < d->num_candidates; c += d->num_threads) {

        long jlo = c * CHUNK;
        long jhi = jlo + CHUNK;

        if (jhi > d->num_candidates) {
            jhi = d->num_candidates;
        }

        for (long j = jlo; j < jhi; j++) {

            int k = 3 + 2 * (int) j;

            if (is_prime(k)) {
                d->flags[k] = 1;
            }
        }
    }

    d->search_time = now_seconds() - t0;

    /* Every flag must be written before anyone starts reading them. */
    pthread_barrier_wait(d->barrier);

    /* ---- Phase A: count set flags in this thread's contiguous block -----
     * The block covers candidate values k, not chunk indices, so that the
     * output written in Phase C comes out already sorted. */
    double t1 = now_seconds();

    long span = (long) d->n - 2;
    long klo = 2 + span * d->id / d->num_threads;
    long khi = 2 + span * (d->id + 1) / d->num_threads;

    int local_count = 0;
    for (long k = klo; k < khi; k++) {
        if (d->flags[k]) {
            local_count++;
        }
    }

    /* offsets[0] stays 0; thread i publishes its count at offsets[i+1]. */
    d->offsets[d->id + 1] = local_count;

    /* ---- Phase B: exclusive prefix sum, done by exactly one thread ------
     * pthread_barrier_wait returns PTHREAD_BARRIER_SERIAL_THREAD in exactly
     * one of the waiting threads, which is a convenient way to elect a
     * leader without an extra mutex. O(p) work, p = number of threads. */
    int rc = pthread_barrier_wait(d->barrier);

    if (rc == PTHREAD_BARRIER_SERIAL_THREAD) {
        for (int i = 0; i < d->num_threads; i++) {
            d->offsets[i + 1] += d->offsets[i];
        }
    }

    /* Nobody may read offsets[] until the leader has finished writing it. */
    pthread_barrier_wait(d->barrier);

    /* ---- Phase C: write this block's primes at the computed offset ------ */
    int idx = d->offsets[d->id];
    for (long k = klo; k < khi; k++) {
        if (d->flags[k]) {
            d->primes[idx++] = (int) k;
        }
    }

    d->compact_time = now_seconds() - t1;

    return NULL;
}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("Usage: %s <n> <number_of_threads>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    if (n <= 2) {
        printf("There are no prime numbers strictly less than %d.\n", n);
        return 0;
    }

    if (num_threads < 1) {
        printf("Number of threads must be at least 1.\n");
        return 1;
    }

    double total_start = now_seconds();

    /* Buffers are allocated outside the timed region, matching task1.c. */
    char *flags        = calloc((size_t) n, sizeof(char));
    pthread_t *threads = malloc((size_t) num_threads * sizeof(pthread_t));
    ThreadData *thread_data = malloc((size_t) num_threads * sizeof(ThreadData));
    int *offsets       = calloc((size_t) num_threads + 1, sizeof(int));

    long capacity = prime_count_bound(n);
    int *primes   = malloc((size_t) capacity * sizeof(int));

    if (flags == NULL || threads == NULL || thread_data == NULL
        || offsets == NULL || primes == NULL) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        free(flags);
        free(threads);
        free(thread_data);
        free(offsets);
        free(primes);
        return 1;
    }

    pthread_barrier_t barrier;
    if (pthread_barrier_init(&barrier, NULL, (unsigned) num_threads) != 0) {
        fprintf(stderr, "Error: could not initialise barrier.\n");
        free(flags);
        free(threads);
        free(thread_data);
        free(offsets);
        free(primes);
        return 1;
    }

    /* Odd numbers from 3 up to (but not including) n, counted as an index
     * space 0..num_candidates-1 so chunk boundaries are plain integers;
     * candidate j corresponds to k = 3 + 2*j. */
    long num_candidates = ((long) n - 2) / 2;

    /* ---- Timed region: search + compaction ------------------------------ */
    double compute_start = now_seconds();

    /* 2 is the only even prime and is never visited by the search loop. */
    flags[2] = 1;

    for (int i = 0; i < num_threads; i++) {

        thread_data[i].id             = i;
        thread_data[i].num_threads    = num_threads;
        thread_data[i].n              = n;
        thread_data[i].flags          = flags;
        thread_data[i].primes         = primes;
        thread_data[i].offsets        = offsets;
        thread_data[i].num_candidates = num_candidates;
        thread_data[i].barrier        = &barrier;
        thread_data[i].search_time    = 0.0;
        thread_data[i].compact_time   = 0.0;

        if (pthread_create(&threads[i], NULL, find_primes, &thread_data[i]) != 0) {
            fprintf(stderr, "Error: could not create thread %d.\n", i);
            /* Threads already created are waiting on a barrier that will
             * never be satisfied, so exit rather than attempting to join. */
            exit(1);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    double compute_seconds = now_seconds() - compute_start;
    /* ---- End timed region: sorted list now exists in primes[] ----------- */

    int count = offsets[num_threads];

    /* ---- Output --------------------------------------------------------- */
    if (n < 100) {

        for (int i = 0; i < count; i++) {
            printf("%d ", primes[i]);
        }
        printf("\n");

    } else {

        FILE *fp = fopen("output.txt", "w");

        if (fp == NULL) {
            fprintf(stderr, "Error: could not open output.txt.\n");
            pthread_barrier_destroy(&barrier);
            free(primes);
            free(flags);
            free(threads);
            free(thread_data);
            free(offsets);
            return 1;
        }

        for (int i = 0; i < count; i++) {
            fprintf(fp, "%d\n", primes[i]);
        }

        fclose(fp);
    }

    double total_seconds = now_seconds() - total_start;

    /* ---- Load-balance report --------------------------------------------
     * If the partitioning is balanced, min and max should be close together.
     * A large spread means threads finished at very different times and the
     * early finishers sat idle at the barrier. Search and compaction are
     * reported separately because they use different partitioning schemes:
     * cyclic chunks for the search, contiguous blocks for the compaction. */
    double min_search = thread_data[0].search_time;
    double max_search = thread_data[0].search_time;
    double max_compact = thread_data[0].compact_time;

    for (int i = 1; i < num_threads; i++) {

        if (thread_data[i].search_time < min_search) {
            min_search = thread_data[i].search_time;
        }
        if (thread_data[i].search_time > max_search) {
            max_search = thread_data[i].search_time;
        }
        if (thread_data[i].compact_time > max_compact) {
            max_compact = thread_data[i].compact_time;
        }
    }

    double imbalance = 0.0;
    if (max_search > 0.0) {
        imbalance = 100.0 * (max_search - min_search) / max_search;
    }

    printf("Primes found: %d\n", count);
    printf("Computation time: %.6f seconds\n", compute_seconds);
    printf("Total time (incl. I/O): %.6f seconds\n", total_seconds);
    printf("Search time min: %.6f seconds\n", min_search);
    printf("Search time max: %.6f seconds\n", max_search);
    printf("Search imbalance: %.2f %%\n", imbalance);
    printf("Compaction time (slowest thread): %.6f seconds\n", max_compact);

    pthread_barrier_destroy(&barrier);
    free(primes);
    free(flags);
    free(threads);
    free(thread_data);
    free(offsets);

    return 0;
}