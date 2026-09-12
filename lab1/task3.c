/*
 * Task 3 - OpenMP: Finding Prime Numbers
 * FIT3143 Lab 1 (Week 4)
 *
 * Parallel version of task1 using OpenMP to find primes less than n.
 * Outputs to the console if n < 100, or to output.txt if n >= 100.
 *
 * How it works:
 * - Instead of using locks, each thread updates its own specific index in 
 *   an array (is_prime_flag). This prevents race conditions.
 * - After the parallel section finishes, a normal serial loop goes through 
 *   the array to collect all the primes in order.
 * - I used schedule(dynamic, 1000) because larger numbers take more math 
 *   to check. Dynamic scheduling gives out chunks of work as threads become 
 *   free, which balances the load better than a static split.
 *
 * Compile: gcc task3.c -o task3 -fopenmp -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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
    if (is_prime_flag == NULL) {
        fprintf(stderr, "Error: failed memory allocation.\n");
        return 1;
    }

    omp_set_num_threads(num_threads);

    double start_time = omp_get_wtime();

    is_prime_flag[2] = 1;
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int k = 3; k < n; k += 2) {
        if (is_prime(k)) {
            is_prime_flag[k] = 1;
        }
    }

    double elapsed_seconds = omp_get_wtime() - start_time;
    if (elapsed_seconds < 0.0) {
        elapsed_seconds = 0.0;
    }

    /* Compact the flags into a sorted list (serial, O(n)). */
    int *primes = malloc((size_t) n * sizeof(int));
    if (primes == NULL) {
        fprintf(stderr, "Error: failed memory allocation.\n");
        free(is_prime_flag);
        return 1;
    }
    int count = 0;
    for (int k = 2; k < n; k++) {
        if (is_prime_flag[k]) {
            primes[count++] = k;
        }
    }


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
            return 1;
        }
        for (int i = 0; i < count; i++) {
            fprintf(fp, "%d\n", primes[i]);
        }
        fclose(fp);
    }

    printf("Time taken: %f seconds\n", elapsed_seconds);

    free(primes);
    free(is_prime_flag);
    return 0;
}
