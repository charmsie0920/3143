/*
 * Task 2 - POSIX Threads: Finding Prime Numbers
 * FIT3143 Lab 1 (Week 4)
 *
 * Parallel version of Task 1 using POSIX Threads.
 * - Prints to stdout when n < 100
 * - Writes to a text file (output.txt) when n >= 100
 * - Reports execution time
 *
 * Parallel partitioning scheme:
 * Odd candidates are grouped into fixed-size chunks of CHUNK numbers.
 * Thread `id` statically claims chunks id, id + num_threads, id + 2 *
 * num_threads, ... (a cyclic / round-robin assignment) instead of one
 * contiguous block each. Candidate cost grows with k (more trial divisions
 * up to sqrt(k)), so a naive equal split of the range would give the thread
 * handling the largest k values far more work than the one handling the
 * smallest. Interleaving chunks means every thread gets a roughly even mix
 * of cheap (small k) and expensive (large k) chunks, without needing a
 * shared counter, a lock, or any runtime coordination.
 *
 * Each thread writes primality results straight into a shared flags[]
 * array, one byte per candidate. Because chunk assignment is disjoint,
 * every index is written by exactly one thread, so no synchronisation is
 * needed there either. A single serial pass afterwards compacts flags[]
 * into a sorted primes[] array -- the same technique Task 3 uses with
 * OpenMP, which keeps the two implementations directly comparable.
 *
 * Compile:
 * gcc task2.c -o task2 -pthread -lm
 *
 * Run:
 * ./task2 <n> <number_of_threads>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

/* Odd candidates per chunk. Matches Task 3's schedule(dynamic, 1000) so the
 * two parallel schemes are directly comparable. */
#define CHUNK 1000

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

/* Per-thread state. flags is shared (each thread only ever writes indices
 * it owns); every other field is private. thread_time records how long
 * this thread spent working, used afterwards to report load imbalance. */
typedef struct {
    int id;
    int num_threads;
    char *flags;
    long num_candidates;
    double thread_time;
} ThreadData;


void *find_primes(void *arg) {
    ThreadData *d = (ThreadData *) arg;

    struct timespec t0;
    struct timespec t1;

    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Chunk c belongs to this thread iff c % num_threads == id -- the
     * cyclic assignment described in the file header. */
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

    clock_gettime(CLOCK_MONOTONIC, &t1);

    d->thread_time = (t1.tv_sec - t0.tv_sec)
                   + (t1.tv_nsec - t0.tv_nsec) / 1e9;

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

    struct timespec total_start;
    struct timespec total_end;

    clock_gettime(CLOCK_MONOTONIC, &total_start);

    char *flags = calloc((size_t) n, sizeof(char));

    pthread_t *threads = malloc((size_t) num_threads * sizeof(pthread_t));

    ThreadData *thread_data = malloc((size_t) num_threads * sizeof(ThreadData));

    if (flags == NULL || threads == NULL || thread_data == NULL) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        free(flags);
        free(threads);
        free(thread_data);
        return 1;
    }

    /* Odd numbers from 3 up to (but not including) n, counted as an index
     * space 0..num_candidates-1 so chunk boundaries are plain integers;
     * candidate j corresponds to k = 3 + 2*j. */
    long num_candidates = ((long) n - 2) / 2;

    /* ---- Computation phase --------------------------------------------- */
    struct timespec compute_start;
    struct timespec compute_end;

    clock_gettime(CLOCK_MONOTONIC, &compute_start);

    for (int i = 0; i < num_threads; i++) {

        thread_data[i].id             = i;
        thread_data[i].num_threads    = num_threads;
        thread_data[i].flags          = flags;
        thread_data[i].num_candidates = num_candidates;
        thread_data[i].thread_time    = 0.0;

        if (pthread_create(&threads[i], NULL, find_primes, &thread_data[i]) != 0) {
            fprintf(stderr, "Error: could not create thread %d.\n", i);
            free(flags);
            free(threads);
            free(thread_data);
            return 1;
        }
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &compute_end);

    double compute_seconds =
        (compute_end.tv_sec - compute_start.tv_sec) +
        (compute_end.tv_nsec - compute_start.tv_nsec) / 1e9;

    flags[2] = 1;

    int count = 0;
    for (int k = 2; k < n; k++) {
        if (flags[k]) {
            count++;
        }
    }

    int *primes = malloc((size_t) count * sizeof(int));

    if (primes == NULL) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        free(flags);
        free(threads);
        free(thread_data);
        return 1;
    }

    int idx = 0;
    for (int k = 2; k < n; k++) {
        if (flags[k]) {
            primes[idx++] = k;
        }
    }

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
            free(primes);
            free(flags);
            free(threads);
            free(thread_data);
            return 1;
        }

        for (int i = 0; i < count; i++) {
            fprintf(fp, "%d\n", primes[i]);
        }

        fclose(fp);
    }

    clock_gettime(CLOCK_MONOTONIC, &total_end);

    double total_seconds =
        (total_end.tv_sec - total_start.tv_sec) +
        (total_end.tv_nsec - total_start.tv_nsec) / 1e9;

    /* ---- Load-balance report --------------------------------------------
     * If the partitioning is balanced, min and max should be close together.
     * A large spread means threads finished at very different times and the
     * early finishers sat idle at the join. */
    double min_thread_time = thread_data[0].thread_time;
    double max_thread_time = thread_data[0].thread_time;

    for (int i = 1; i < num_threads; i++) {

        if (thread_data[i].thread_time < min_thread_time) {
            min_thread_time = thread_data[i].thread_time;
        }
        if (thread_data[i].thread_time > max_thread_time) {
            max_thread_time = thread_data[i].thread_time;
        }
    }

    double imbalance = 0.0;
    if (max_thread_time > 0.0) {
        imbalance = 100.0 * (max_thread_time - min_thread_time) / max_thread_time;
    }

    printf("Computation time: %.6f seconds\n", compute_seconds);
    printf("Total time: %.6f seconds\n", total_seconds);
    printf("Thread time min: %.6f seconds\n", min_thread_time);
    printf("Thread time max: %.6f seconds\n", max_thread_time);
    printf("Thread imbalance: %.2f %%\n", imbalance);

    free(primes);
    free(flags);
    free(threads);
    free(thread_data);

    return 0;
}