/*
 * Task 2 - POSIX Threads: Finding Prime Numbers
 * FIT3143 Lab 1 (Week 4)   [REVISED for Lab 2 / Week 8]
 *
 * Parallel version of Task 1 using POSIX Threads.
 *
 * How it works:
 * - Search: threads take fixed-size chunks round-robin (cyclic). Cost per
 *   candidate grows with k, so cyclic assignment gives every thread a fair
 *   mix of cheap and expensive candidates without needing locks.
 * - Compaction: three-phase parallel prefix sum (count block / prefix / fill),
 *   replacing the serial O(n) flag scan flagged in the Lab 1 feedback.
 * - Threads are created once and reused across all phases via a barrier.
 *
 * ---- Phase measurement for Amdahl's Law ------------------------------------
 * The timed region is broken into three categories so the serial fraction
 * can be measured empirically rather than guessed:
 *
 *   parallel_s  max over threads of (search + count + fill)
 *               -- work that divides by the thread count
 *   serial_s    the prefix sum over per-thread counts, plus the file write
 *               -- work that does not shrink as threads are added
 *   overhead_s  total - parallel_s - serial_s
 *               -- thread creation, joins, and time lost waiting at barriers
 *
 * The timed region runs from thread creation until the sorted list has been
 * output (console if n < 100, otherwise output.txt), the same definition as
 * every other version.
 *
 * f = serial_s / total_s is Amdahl's serial fraction. It will be very small
 * here; the interesting quantity is overhead_s, which Amdahl does not model
 * and which is what actually limits the measured speedup.
 *
 * Compile: gcc task2.c -o task2 -O2 -pthread -lm
 * Run:     ./task2 <n> <number_of_threads>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define CHUNK 1000

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec / 1e9;
}

static long prime_count_bound(long x) {
    if (x < 100) {
        return x;
    }
    return (long) (1.25506 * (double) x / log((double) x)) + 1;
}

/* Write primes one per line to path. Formats the digits by hand into one
 * buffer and issues a single fwrite, instead of one fprintf per prime.
 * IDENTICAL to write_primes() in task1_mpi.c and the hybrid task2.c, so the
 * file write costs the same in every version. Returns 0 on success. */
static int write_primes(const char *path, const int *primes, int count) {

    /* A positive int has at most 10 digits, plus the newline. */
    char *buf = malloc((size_t) count * 11 + 1);
    if (buf == NULL) {
        return -1;
    }

    char *p = buf;
    for (int i = 0; i < count; i++) {
        char     digits[10];
        int      len = 0;
        unsigned v   = (unsigned) primes[i];
        do {
            digits[len++] = (char) ('0' + v % 10);
            v /= 10;
        } while (v > 0);
        while (len > 0) {
            *p++ = digits[--len];
        }
        *p++ = '\n';
    }

    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        free(buf);
        return -1;
    }

    size_t bytes   = (size_t) (p - buf);
    size_t written = fwrite(buf, 1, bytes, fp);
    int    closed  = fclose(fp);

    free(buf);
    return (written == bytes && closed == 0) ? 0 : -1;
}

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
    double prefix_time;   /* nonzero only for the thread that runs Phase B */
} ThreadData;

void *find_primes(void *arg) {
    ThreadData *d = (ThreadData *) arg;

    /* ---- Search: cyclic chunk partitioning ------------------------------ */
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

    pthread_barrier_wait(d->barrier);

    /* ---- Phase A: count set flags in this thread's contiguous block ----- */
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
    d->offsets[d->id + 1] = local_count;

    double t_countdone = now_seconds();

    /* ---- Phase B: exclusive prefix sum, one elected thread -------------- */
    int rc = pthread_barrier_wait(d->barrier);

    double t_prefix0 = now_seconds();
    if (rc == PTHREAD_BARRIER_SERIAL_THREAD) {
        for (int i = 0; i < d->num_threads; i++) {
            d->offsets[i + 1] += d->offsets[i];
        }
        d->prefix_time = now_seconds() - t_prefix0;
    }

    pthread_barrier_wait(d->barrier);

    /* ---- Phase C: write this block's primes at the computed offset ------ */
    double t_fill0 = now_seconds();

    int idx = d->offsets[d->id];
    for (long k = klo; k < khi; k++) {
        if (d->flags[k]) {
            d->primes[idx++] = (int) k;
        }
    }

    /* Count and fill are this thread's own scalable work; the barrier waits
     * between them are overhead and are deliberately excluded. */
    d->compact_time = (t_countdone - t1) + (now_seconds() - t_fill0);

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

    char *flags        = calloc((size_t) n, sizeof(char));
    pthread_t *threads = malloc((size_t) num_threads * sizeof(pthread_t));
    ThreadData *thread_data = malloc((size_t) num_threads * sizeof(ThreadData));
    int *offsets       = calloc((size_t) num_threads + 1, sizeof(int));

    long capacity = prime_count_bound((long) n);
    int *primes   = malloc((size_t) capacity * sizeof(int));

    if (flags == NULL || threads == NULL || thread_data == NULL
        || offsets == NULL || primes == NULL) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        return 1;
    }

    pthread_barrier_t barrier;
    if (pthread_barrier_init(&barrier, NULL, (unsigned) num_threads) != 0) {
        fprintf(stderr, "Error: could not initialise barrier.\n");
        return 1;
    }

    long num_candidates = ((long) n - 2) / 2;

    /* ---- Timed region: search + compaction + file write ----------------- */
    double compute_start = now_seconds();

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
        thread_data[i].prefix_time    = 0.0;

        if (pthread_create(&threads[i], NULL, find_primes, &thread_data[i]) != 0) {
            fprintf(stderr, "Error: could not create thread %d.\n", i);
            exit(1);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    double compute_seconds = now_seconds() - compute_start;
    /* ---- Compaction done; the file write below is also timed ------------ */

    int count = offsets[num_threads];

    /* ---- Output: timed, and added to the total -------------------------
     * As in Lab 1: n < 100 prints to the console, otherwise the primes are
     * written to output.txt. Output is serial work in every version, so it
     * is reported on its own as write_s and counted in the serial part --
     * exactly as in task1_mpi.c and the hybrid task2.c. */
    double t_write0 = now_seconds();

    if (n < 100) {
        for (int i = 0; i < count; i++) {
            printf("%d ", primes[i]);
        }
        printf("\n");
    } else if (write_primes("output.txt", primes, count) != 0) {
        fprintf(stderr, "Error: could not write output.txt.\n");
        pthread_barrier_destroy(&barrier);
        free(primes);
        free(flags);
        free(threads);
        free(thread_data);
        free(offsets);
        return 1;
    }

    double t_write = now_seconds() - t_write0;
    double total_seconds = compute_seconds + t_write;

    /* ---- Phase accounting ------------------------------------------------ */
    double min_search  = thread_data[0].search_time;
    double max_search  = thread_data[0].search_time;
    double max_work    = 0.0;
    double prefix_time = 0.0;

    for (int i = 0; i < num_threads; i++) {

        double work = thread_data[i].search_time + thread_data[i].compact_time;
        if (work > max_work) {
            max_work = work;
        }
        if (thread_data[i].search_time < min_search) {
            min_search = thread_data[i].search_time;
        }
        if (thread_data[i].search_time > max_search) {
            max_search = thread_data[i].search_time;
        }
        if (thread_data[i].prefix_time > prefix_time) {
            prefix_time = thread_data[i].prefix_time;
        }
    }

    double imbalance = 0.0;
    if (max_search > 0.0) {
        imbalance = 100.0 * (max_search - min_search) / max_search;
    }

    double serial_part = prefix_time + t_write;
    double overhead    = total_seconds - max_work - serial_part;
    if (overhead < 0.0) {
        overhead = 0.0;
    }

    printf("Primes found: %d\n", count);
    printf("Total time (incl. file write): %.6f seconds\n", total_seconds);
    printf("  parallel (slowest thread): %.6f s\n", max_work);
    printf("  serial (prefix + write)  : %.6f s\n", serial_part);
    printf("    prefix sum             : %.6f s\n", prefix_time);
    printf("    file write             : %.6f s\n", t_write);
    printf("  overhead (create/join)   : %.6f s\n", overhead);
    printf("  search imbalance         : %.2f %%\n", imbalance);

    /* impl,scheme,n,procs,threads,nodes,primes,
     * total,serial,parallel,overhead,imbalance,write */
    printf("CSV,pthreads,cyclic,%d,1,%d,1,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.6f\n",
           n, num_threads, count, total_seconds, serial_part, max_work,
           overhead, imbalance, t_write);

    pthread_barrier_destroy(&barrier);
    free(primes);
    free(flags);
    free(threads);
    free(thread_data);
    free(offsets);

    return 0;
}