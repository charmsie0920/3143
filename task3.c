/*
 * Task 3 - OpenMP: Finding Prime Numbers
 * FIT3143 Lab 1 (Week 4)
 *
 * Parallel version of task1.c using OpenMP.
 * Finds all prime numbers strictly less than a user-provided integer n.
 * - Prints to stdout when n < 100
 * - Writes to a text file (output.txt) when n >= 100
 * - Reports execution time (wall clock, via omp_get_wtime)
 *
 * Parallel partitioning scheme:
 * Each thread tests a subset of odd candidates k and writes its primality
 * result directly to a unique slot (is_prime_flag[k]). Because each thread
 * only ever writes to indices it owns, there is no race condition and no
 * locking is needed. A single serial pass afterwards compacts the flags
 * into a sorted primes[] array (the k values are scanned in increasing
 * order, so the result comes out sorted for free).
 *
 * schedule(dynamic, 1000) is used because candidate cost grows with k
 * (more trial divisions up to sqrt(k)): an equal static split of the range
 * would give the thread handling the largest k values more work than the
 * thread handling the smallest ones. Dynamic scheduling hands out chunks
 * on demand so faster threads pick up more chunks, balancing the load.
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
