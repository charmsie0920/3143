/*
 * Task 3 - OpenMP: Finding Prime Numbers
 * FIT3143 Lab 1 (Week 4)   [REVISED for Lab 2 / Week 8]
 *
 * Parallel version of task1 using OpenMP to find primes less than n.
 * Outputs to the console if n < 100, or to output.txt if n >= 100.
 *
 * How it works:
 * - Instead of using locks, each thread updates its own specific index in
 *   an array (is_prime_flag). This prevents race conditions.
 * - schedule(dynamic, 1000) is used because larger numbers take more math
 *   to check. Dynamic scheduling gives out chunks of work as threads become
 *   free, which balances the load better than a static split.
 * - The flags are then compacted into a sorted list, IN PARALLEL (see below).
 *
 * ---- Revisions for Lab 2 (see CHANGES.md) ----------------------------------
 * 1. The serial O(n) flag-to-array pass is gone. Marker feedback: "has
 *    additional O(n) serial flag to array pass". Compaction is now parallel,
 *    using the same three-phase prefix-sum scheme as the revised task2.c:
 *
 *      Phase A  each thread counts set flags in its own CONTIGUOUS block
 *      Phase B  #pragma omp single computes the exclusive prefix sum,
 *               giving each thread its write offset (O(p) work)
 *      Phase C  each thread copies its block's primes to primes[offset...]
 *
 *    Output is sorted without a sort step, because the blocks are contiguous
 *    and ascending and each thread scans its block in ascending order.
 *
 *    Everything now sits inside ONE `#pragma omp parallel` region, so the
 *    threads are not torn down and respawned between the search and the
 *    compaction. The implicit barrier at the end of `omp for` guarantees all
 *    flags are written before any thread starts counting.
 *
 * 2. Timing: the timed region now covers search AND compaction, ending when
 *    the sorted list exists in memory. This matches task1.c and task2.c
 *    exactly, so the reported speedups compare like with like.
 *
 * 3. Output buffer sized by the Rosser-Schoenfeld bound rather than
 *    allocating n ints (40 MB -> ~3 MB at n = 10,000,000).
 *
 * 4. omp_get_wtime() replaced with clock_gettime(CLOCK_MONOTONIC) so all
 *    four versions use one identical clock. (omp_get_wtime is also
 *    wall-clock, so this is for consistency rather than correctness, and it
 *    removes the need for the old negative-elapsed-time guard.)
 *
 * Compile: gcc task3.c -o task3 -O2 -fopenmp -lm
 * Run:     ./task3 <n> <number_of_threads>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

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

int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("Usage: %s <n> <number_of_threads>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    if (n <= 2) {
        printf("no prime numbers less than %d.\n", n);
        return 0;
    }

    if (num_threads < 1) {
        printf("at least 1 thread required.\n");
        return 1;
    }

    /* One flag per candidate; each thread owns disjoint indices, so writes
     * need no synchronisation. calloc zero-initialises to "not prime". */
    char *is_prime_flag = calloc((size_t) n, sizeof(char));

    /* offsets[i] is where thread i's primes begin in the output array. */
    int *offsets = calloc((size_t) num_threads + 1, sizeof(int));

    long capacity = prime_count_bound(n);
    int *primes = malloc((size_t) capacity * sizeof(int));

    if (is_prime_flag == NULL || offsets == NULL || primes == NULL) {
        fprintf(stderr, "Error: failed memory allocation.\n");
        free(is_prime_flag);
        free(offsets);
        free(primes);
        return 1;
    }

    omp_set_num_threads(num_threads);

    /* The runtime may hand back fewer threads than requested; the actual
     * count is captured inside the region and used to read the final total. */
    int threads_used = num_threads;

    /* ---- Timed region: search + compaction ------------------------------ */
    double start_time = now_seconds();

    is_prime_flag[2] = 1;

    #pragma omp parallel
    {
        /* ---- Search phase: dynamic chunks of 1000 odd candidates ------- */
        #pragma omp for schedule(dynamic, 1000)
        for (int k = 3; k < n; k += 2) {
            if (is_prime(k)) {
                is_prime_flag[k] = 1;
            }
        }
        /* Implicit barrier here: every flag is written before Phase A. */

        int id = omp_get_thread_num();
        int nt = omp_get_num_threads();

        /* Contiguous block of candidate values owned by this thread. Blocks
         * are used here (rather than the dynamic chunks of the search) so
         * that Phase C produces sorted output with no sort step. */
        long span = (long) n - 2;
        long klo = 2 + span * id / nt;
        long khi = 2 + span * (id + 1) / nt;

        /* ---- Phase A: count set flags in this block -------------------- */
        int local_count = 0;
        for (long k = klo; k < khi; k++) {
            if (is_prime_flag[k]) {
                local_count++;
            }
        }
        offsets[id + 1] = local_count;

        #pragma omp barrier

        /* ---- Phase B: exclusive prefix sum, O(p) work ------------------ */
        #pragma omp single
        {
            for (int i = 0; i < nt; i++) {
                offsets[i + 1] += offsets[i];
            }
            threads_used = nt;
        }
        /* Implicit barrier at the end of `single`: offsets[] is now safe
         * for every thread to read. */

        /* ---- Phase C: write this block's primes at its own offset ------ */
        int idx = offsets[id];
        for (long k = klo; k < khi; k++) {
            if (is_prime_flag[k]) {
                primes[idx++] = (int) k;
            }
        }
    }

    double elapsed_seconds = now_seconds() - start_time;
    /* ---- End timed region: sorted list now exists in primes[] ----------- */

    int count = offsets[threads_used];

    if (n < 100) {
        for (int i = 0; i < count; i++) {
            printf("%d ", primes[i]);
        }
        printf("\n");
    } else {
        FILE *fp = fopen("output.txt", "w");
        if (fp == NULL) {
            fprintf(stderr, "Error: no output file\n");
            free(primes);
            free(is_prime_flag);
            free(offsets);
            return 1;
        }
        for (int i = 0; i < count; i++) {
            fprintf(fp, "%d\n", primes[i]);
        }
        fclose(fp);
    }

    printf("Primes found: %d\n", count);
    printf("Threads used: %d\n", threads_used);
    printf("Time taken: %.6f seconds\n", elapsed_seconds);

    free(primes);
    free(is_prime_flag);
    free(offsets);
    return 0;
}