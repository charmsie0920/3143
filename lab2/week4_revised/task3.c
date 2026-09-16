/*
 * Task 3 - OpenMP: Finding Prime Numbers
 * FIT3143 Lab 1 (Week 4)   [REVISED for Lab 2 / Week 8]
 *
 * Parallel version of Task 1 using OpenMP.
 *
 * How it works:
 * - Search: schedule(dynamic, 1000). Larger numbers cost more to test, so
 *   dynamic scheduling hands out chunks as threads become free, balancing
 *   better than a static split.
 * - Compaction: three-phase parallel prefix sum (count block / prefix / fill),
 *   replacing the serial O(n) flag-to-array pass flagged in the feedback.
 * - Everything sits in ONE parallel region, so threads are not torn down and
 *   respawned between the search and the compaction.
 *
 * ---- Phase measurement for Amdahl's Law ------------------------------------
 *   parallel_s  max over threads of (search + count + fill)
 *   serial_s    the prefix sum (one thread, O(p)) plus the file write
 *   overhead_s  total - parallel_s - serial_s (region entry/exit, barriers)
 *
 * The timed region runs from entering the parallel region until the sorted
 * list has been output (console if n < 100, otherwise output.txt), the
 * same definition as every other version.
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

    char *is_prime_flag = calloc((size_t) n, sizeof(char));
    int  *offsets       = calloc((size_t) num_threads + 1, sizeof(int));

    long capacity = prime_count_bound((long) n);
    int *primes = malloc((size_t) capacity * sizeof(int));

    /* Per-thread phase timings, collected inside the parallel region. */
    double *t_search  = calloc((size_t) num_threads, sizeof(double));
    double *t_compact = calloc((size_t) num_threads, sizeof(double));

    if (is_prime_flag == NULL || offsets == NULL || primes == NULL
        || t_search == NULL || t_compact == NULL) {
        fprintf(stderr, "Error: failed memory allocation.\n");
        return 1;
    }

    omp_set_num_threads(num_threads);

    int    threads_used = num_threads;
    double prefix_time  = 0.0;

    /* ---- Timed region: search + compaction + file write ----------------- */
    double start_time = now_seconds();

    is_prime_flag[2] = 1;

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nt = omp_get_num_threads();

        double s0 = now_seconds();

        #pragma omp for schedule(dynamic, 1000)
        for (int k = 3; k < n; k += 2) {
            if (is_prime(k)) {
                is_prime_flag[k] = 1;
            }
        }
        /* Implicit barrier: all flags written before Phase A. */

        t_search[id] = now_seconds() - s0;

        long span = (long) n - 2;
        long klo = 2 + span * id / nt;
        long khi = 2 + span * (id + 1) / nt;

        /* ---- Phase A: count ---------------------------------------------- */
        double c0 = now_seconds();

        int local_count = 0;
        for (long k = klo; k < khi; k++) {
            if (is_prime_flag[k]) {
                local_count++;
            }
        }
        offsets[id + 1] = local_count;

        double c1 = now_seconds();

        #pragma omp barrier

        /* ---- Phase B: prefix sum ----------------------------------------- */
        #pragma omp single
        {
            double p0 = now_seconds();
            for (int i = 0; i < nt; i++) {
                offsets[i + 1] += offsets[i];
            }
            prefix_time  = now_seconds() - p0;
            threads_used = nt;
        }
        /* Implicit barrier after single. */

        /* ---- Phase C: fill ------------------------------------------------ */
        double f0 = now_seconds();

        int idx = offsets[id];
        for (long k = klo; k < khi; k++) {
            if (is_prime_flag[k]) {
                primes[idx++] = (int) k;
            }
        }

        /* Barrier waits between the phases are overhead, not this thread's
         * own scalable work, so they are excluded here. */
        t_compact[id] = (c1 - c0) + (now_seconds() - f0);
    }

    double elapsed_seconds = now_seconds() - start_time;
    /* ---- Compaction done; the file write below is also timed ------------ */

    int count = offsets[threads_used];

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
        free(primes);
        free(is_prime_flag);
        free(offsets);
        free(t_search);
        free(t_compact);
        return 1;
    }

    double t_write = now_seconds() - t_write0;
    double total_seconds = elapsed_seconds + t_write;

    /* ---- Phase accounting ------------------------------------------------ */
    double min_search = t_search[0];
    double max_search = t_search[0];
    double max_work   = 0.0;

    for (int i = 0; i < threads_used; i++) {
        double work = t_search[i] + t_compact[i];
        if (work > max_work)         max_work   = work;
        if (t_search[i] < min_search) min_search = t_search[i];
        if (t_search[i] > max_search) max_search = t_search[i];
    }

    /* Note: with schedule(dynamic) the implicit barrier at the end of the
     * omp for means every thread's measured search time includes any wait
     * for stragglers, so this imbalance figure understates the true spread.
     * It is reported for consistency with the pthreads version. */
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
    printf("Threads used: %d\n", threads_used);
    printf("Total time (incl. file write): %.6f seconds\n", total_seconds);
    printf("  parallel (slowest thread): %.6f s\n", max_work);
    printf("  serial (prefix + write)  : %.6f s\n", serial_part);
    printf("    prefix sum             : %.6f s\n", prefix_time);
    printf("    file write             : %.6f s\n", t_write);
    printf("  overhead (region/barrier): %.6f s\n", overhead);
    printf("  search imbalance         : %.2f %%\n", imbalance);

    /* impl,scheme,n,procs,threads,nodes,primes,
     * total,serial,parallel,overhead,imbalance,write */
    printf("CSV,openmp,dynamic,%d,1,%d,1,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.6f\n",
           n, threads_used, count, total_seconds, serial_part, max_work,
           overhead, imbalance, t_write);

    free(primes);
    free(is_prime_flag);
    free(offsets);
    free(t_search);
    free(t_compact);
    return 0;
}