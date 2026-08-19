/*
 * Task 1 - Serial Code: Finding Prime Numbers
 * FIT3143 Lab 1 (Week 4)
 *
 * Finds all prime numbers strictly less than a user-provided integer n.
 * - Prints to stdout when n < 100
 * - Writes to a text file (output.txt) when n >= 100
 * - Reports execution time
 *
 * Compile: gcc task1.c -o task1 -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

    /* Worst case (small n) needs at most n-2 slots; fine up to n = 10,000,000. */
    int *primes = malloc((size_t) n * sizeof(int));
    if (primes == NULL) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        return 1;
    }

    clock_t start_time = clock();

    /* 2 is the only even prime, so it is added directly and the search
     * loop only walks odd candidates (k += 2), halving the work up front. */
    int count = 0;
    primes[count++] = 2;
    for (int k = 3; k < n; k += 2) {
        if (is_prime(k)) {
            primes[count++] = k;
        }
    }

    clock_t end_time = clock();
    double elapsed_seconds = (double) (end_time - start_time) / CLOCKS_PER_SEC;

    /* Timing stops before output (printing / file I/O), so it measures only
     * the search itself -- consistent with how Task 2 and Task 3 time their
     * parallel region, keeping the speedup comparison fair. */
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

    printf("Time taken: %f seconds\n", elapsed_seconds);

    free(primes);
    return 0;
}
