/*
 * Task 1 - Serial Code: Finding Prime Numbers
 * FIT3143 Lab 1 (Week 4)   [REVISED for Lab 2 / Week 8]
 *
 * Finds all prime numbers strictly less than a user-provided integer n.
 * - Prints to stdout when n < 100
 * - Writes to a text file (output.txt) when n >= 100
 * - Reports execution time
 *
 * ---- Revisions for Lab 2  ----------------------------------
 * 1. Timing changed from clock() to clock_gettime(CLOCK_MONOTONIC).
 *    clock() measures CPU time, while Tasks 2/3 measured wall-clock time, so
 *    the serial baseline was not directly comparable with the parallel runs.
 *    All four versions (serial, pthreads, OpenMP, MPI) now use wall-clock.
 *
 * 2. Timed region redefined consistently across all versions as:
 *       "from the start of the search until the sorted prime list exists
 *        in memory, excluding buffer allocation and file I/O."
 *    Previously Task 1 was charged for building its result array while
 *    Tasks 2/3 stopped their clocks before the equivalent step, which
 *    inflated the measured speedups.
 *
 * 3. Result buffer sized by the Rosser-Schoenfeld bound
 *       pi(n) < 1.25506 * n / ln(n)     (n > 1)
 *    instead of allocating n ints. At n = 10,000,000 this is ~779k slots
 *    (~3 MB) rather than 10,000,000 slots (~40 MB), addressing the marker's
 *    note that the implementation "allocates n integers".
 *
 * The prime test itself is UNCHANGED, so the Week 4 vs Week 8 comparison
 * still measures parallelisation rather than a different algorithm.
 *
 * Compile: gcc task1.c -o task1 -O2 -lm
 * Run:     ./task1 <n>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* Wall-clock seconds from a monotonic source. Monotonic is used rather than
 * CLOCK_REALTIME so the measurement cannot be disturbed by NTP adjustments. */
static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec / 1e9;
}

/* Upper bound on the number of primes below n, used to size the result
 * buffer exactly once instead of over-allocating n slots.
 * Rosser & Schoenfeld (1962): pi(x) < 1.25506 * x / ln(x) for x > 1.
 * For very small n the bound is not worth trusting, so fall back to n. */
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
    if (argc != 2) {
        printf("Usage: %s <n>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);

    if (n <= 2) {
        printf("There are no prime numbers strictly less than %d.\n", n);
        return 0;
    }

    /* Allocated before the clock starts, matching Tasks 2 and 3 where the
     * flag array is also allocated outside the timed region. */
    long capacity = prime_count_bound(n);
    int *primes = malloc((size_t) capacity * sizeof(int));
    if (primes == NULL) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        return 1;
    }

    /* ---- Timed region --------------------------------------------------- */
    double start_time = now_seconds();

    /* 2 is the only even prime, so it is added directly and the search
     * loop only walks odd candidates (k += 2), halving the work up front. */
    int count = 0;
    primes[count++] = 2;
    for (int k = 3; k < n; k += 2) {
        if (is_prime(k)) {
            primes[count++] = k;
        }
    }

    double elapsed_seconds = now_seconds() - start_time;
    /* ---- End timed region ------------------------------------------------
     * The sorted list now exists in memory. File I/O below is excluded, as
     * it is identical in every version and is not part of the computation. */

    if (n < 100) {
        for (int i = 0; i < count; i++) {
            printf("%d ", primes[i]);
        }
        printf("\n");
    } else {
        FILE *fp = fopen("output.txt", "w");
        if (fp == NULL) {
            fprintf(stderr, "Error: could not open output file.\n");
            free(primes);
            return 1;
        }
        for (int i = 0; i < count; i++) {
            fprintf(fp, "%d\n", primes[i]);
        }
        fclose(fp);
    }

    printf("Primes found: %d\n", count);
    printf("Time taken: %.6f seconds\n", elapsed_seconds);

    free(primes);
    return 0;
}