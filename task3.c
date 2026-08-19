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

double elapsed(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
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

    struct timespec total_start;
    struct timespec total_end;
 
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    /* One flag per candidate; each thread owns disjoint indices, so writes
     * need no synchronisation. calloc zero-initialises to "not prime". */
    char *is_prime_flag = calloc((size_t) n, sizeof(char));

    double *thread_times = calloc((size_t) num_threads, sizeof(double));

    if (is_prime_flag == NULL) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        return 1;
    }

    omp_set_num_threads(num_threads);

    int actual_threads = num_threads;

    struct timespec compute_start;
    struct timespec compute_end;
 
    clock_gettime(CLOCK_MONOTONIC, &compute_start);

    is_prime_flag[2] = 1;

    #pragma omp parallel
    {
        struct timespec t0;
        struct timespec t1;
 
        int tid = omp_get_thread_num();
 
        if (tid == 0) {
            actual_threads = omp_get_num_threads();
        }
 
        clock_gettime(CLOCK_MONOTONIC, &t0);


    #pragma omp parallel for schedule(dynamic, 1000)
    for (int k = 3; k < n; k += 2) {
        if (is_prime(k)) {
            is_prime_flag[k] = 1;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
 
    thread_times[tid] = elapsed(t0, t1);
    }

    clock_gettime(CLOCK_MONOTONIC, &compute_end);
 
    double compute_seconds = elapsed(compute_start, compute_end);

    /* Compact the flags into a sorted list (serial, O(n)). */
    int *primes = malloc((size_t) n * sizeof(int));
    if (primes == NULL) {
        fprintf(stderr, "Error: memory allocation failed.\n");
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
            fprintf(stderr, "Error: could not open output file.\n");
            free(primes);
            free(is_prime_flag);
            return 1;
        }
        for (int i = 0; i < count; i++) {
            fprintf(fp, "%d\n", primes[i]);
        }
        fclose(fp);
        printf("Found %d prime numbers less than %d.\n", count, n);
        printf("Output written to output.txt\n");
    }

    clock_gettime(CLOCK_MONOTONIC, &total_end);
 
    double total_seconds = elapsed(total_start, total_end);


    double min_thread_time = thread_times[0];
    double max_thread_time = thread_times[0];
 
    for (int i = 1; i < actual_threads; i++) {
 
        if (thread_times[i] < min_thread_time) {
            min_thread_time = thread_times[i];
        }
        if (thread_times[i] > max_thread_time) {
            max_thread_time = thread_times[i];
        }
    }
 
    double imbalance = 0.0;
    if (max_thread_time > 0.0) {
        imbalance = 100.0 * (max_thread_time - min_thread_time) / max_thread_time;
    }
 
    printf("Number of threads: %d\n", actual_threads);
    printf("Computation time: %.6f seconds\n", compute_seconds);
    printf("Total time: %.6f seconds\n", total_seconds);
    printf("Thread time min: %.6f seconds\n", min_thread_time);
    printf("Thread time max: %.6f seconds\n", max_thread_time);
    printf("Thread imbalance: %.2f %%\n", imbalance);

    free(primes);
    free(is_prime_flag);
    free(thread_times);
    return 0;
}
