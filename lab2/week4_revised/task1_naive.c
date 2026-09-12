/*
 * Task 1 (supplementary) - Naive vs Optimised Serial Comparison
 * FIT3143 Lab 1 (Week 4)   [NEW file, added for Lab 2 / Week 8]
 *
 * Marker feedback on Lab 1 Task 1:
 *   "The implementation skips even candidates and checks divisors up to
 *    sqrt(k), but the optimisation is not experimentally compared with a
 *    basic implementation."
 *
 * This program provides that comparison. All three prime tests live in one
 * binary so they share identical timing code, identical output handling and
 * identical compiler flags -- the only variable is the algorithm itself.
 *
 *   Mode 0  Basic trial division.
 *           Every candidate 2..n-1, every divisor 2..k-1.
 *           Cost per candidate is O(k), so total cost is O(n^2).
 *           Only feasible up to roughly n = 200,000.
 *
 *   Mode 1  Square-root cutoff only.
 *           Every candidate 2..n-1, divisors 2..sqrt(k) (odd and even).
 *           Isolates the benefit of the sqrt cutoff on its own.
 *
 *   Mode 2  Full optimisation (identical to task1.c).
 *           Odd candidates only, odd divisors up to sqrt(k).
 *           Isolates the additional benefit of skipping even candidates
 *           and even divisors on top of the sqrt cutoff.
 *
 * Compile: gcc task1_naive.c -o task1_naive -O2 -lm
 * Run:     ./task1_naive <n> <mode 0|1|2>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

/* Mode 0: basic trial division, no cutoff, no parity shortcuts. */
static int is_prime_basic(int k) {
    if (k < 2) {
        return 0;
    }
    for (int divisor = 2; divisor < k; divisor++) {
        if (k % divisor == 0) {
            return 0;
        }
    }
    return 1;
}

/* Mode 1: square-root cutoff, but every divisor is still tested. */
static int is_prime_sqrt(int k) {
    if (k < 2) {
        return 0;
    }
    int limit = (int) sqrt((double) k);
    for (int divisor = 2; divisor <= limit; divisor++) {
        if (k % divisor == 0) {
            return 0;
        }
    }
    return 1;
}

/* Mode 2: identical to the is_prime() used in task1.c / task2.c / task3.c. */
static int is_prime_optimised(int k) {
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
        printf("Usage: %s <n> <mode>\n", argv[0]);
        printf("  mode 0 = basic trial division (divisors 2..k-1)\n");
        printf("  mode 1 = sqrt cutoff, all divisors\n");
        printf("  mode 2 = sqrt cutoff, odd candidates and odd divisors\n");
        return 1;
    }

    int n = atoi(argv[1]);
    int mode = atoi(argv[2]);

    if (n <= 2) {
        printf("There are no prime numbers strictly less than %d.\n", n);
        return 0;
    }

    if (mode < 0 || mode > 2) {
        printf("Mode must be 0, 1 or 2.\n");
        return 1;
    }

    long capacity = prime_count_bound(n);
    int *primes = malloc((size_t) capacity * sizeof(int));
    if (primes == NULL) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        return 1;
    }

    int count = 0;

    /* ---- Timed region ---------------------------------------------------
     * Same definition as task1.c: search only, buffer already allocated,
     * file I/O excluded. The mode branch sits outside the loops so the
     * comparison is not distorted by a per-candidate switch. */
    double start_time = now_seconds();

    if (mode == 0) {
        for (int k = 2; k < n; k++) {
            if (is_prime_basic(k)) {
                primes[count++] = k;
            }
        }
    } else if (mode == 1) {
        for (int k = 2; k < n; k++) {
            if (is_prime_sqrt(k)) {
                primes[count++] = k;
            }
        }
    } else {
        primes[count++] = 2;
        for (int k = 3; k < n; k += 2) {
            if (is_prime_optimised(k)) {
                primes[count++] = k;
            }
        }
    }

    double elapsed_seconds = now_seconds() - start_time;
    /* ---- End timed region ------------------------------------------------ */

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

    /* CSV-friendly line so runs can be appended straight into a results file. */
    printf("mode=%d n=%d primes=%d time=%.6f\n", mode, n, count, elapsed_seconds);

    free(primes);
    return 0;
}