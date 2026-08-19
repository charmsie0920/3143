/*
 * Task 2 - POSIX Threads: Finding Prime Numbers
 * FIT3143 Lab 1 (Week 4)
 *
 * Parallel version of Task 1 using POSIX Threads.
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


/*
 * Returns 1 if k is prime, 0 otherwise.
 *
 * Same prime-checking function as Task 1.
 */
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


/*
 * Information given to each thread.
 */
typedef struct {

    int start;
    int end;

    int *primes;
    int count;

} ThreadData;


/*
 * Function executed by each POSIX thread.
 *
 * Each thread searches its own range of odd numbers
 * and stores its discovered primes in its own array.
 */
void *find_primes(void *arg) {

    ThreadData *data = (ThreadData *) arg;

    data->count = 0;

    /*
     * Make sure the starting value is odd.
     */
    int start = data->start;

    if (start < 3) {
        start = 3;
    }

    if (start % 2 == 0) {
        start++;
    }


    /*
     * Only check odd numbers, matching Task 1.
     */
    for (int k = start; k < data->end; k += 2) {

        if (is_prime(k)) {

            data->primes[data->count] = k;

            data->count++;
        }
    }

    return NULL;
}


int main(int argc, char *argv[]) {

    /*
     * Command-line arguments:
     *
     * ./task2 <n> <number_of_threads>
     */
    if (argc != 3) {

        printf(
            "Usage: %s <n> <number_of_threads>\n",
            argv[0]
        );

        return 1;
    }


    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);


    if (n <= 2) {

        printf(
            "There are no prime numbers strictly less than %d.\n",
            n
        );

        return 0;
    }


    if (num_threads < 1) {

        printf(
            "Number of threads must be at least 1.\n"
        );

        return 1;
    }


    /*
     * Allocate the POSIX thread handles.
     */
    pthread_t *threads =
        malloc(
            (size_t) num_threads *
            sizeof(pthread_t)
        );


    /*
     * Allocate information for each thread.
     */
    ThreadData *thread_data =
        malloc(
            (size_t) num_threads *
            sizeof(ThreadData)
        );


    if (threads == NULL || thread_data == NULL) {

        fprintf(
            stderr,
            "Error: memory allocation failed.\n"
        );

        free(threads);
        free(thread_data);

        return 1;
    }


    /*
     * Allocate a result array for each thread.
     *
     * Each thread can store at most approximately n/2
     * odd numbers, so n/2 + 1 is sufficient.
     */
    size_t max_primes_per_thread =
        ((size_t) n / 2) + 1;


    for (int i = 0; i < num_threads; i++) {

        thread_data[i].primes =
            malloc(
                max_primes_per_thread *
                sizeof(int)
            );

        if (thread_data[i].primes == NULL) {

            fprintf(
                stderr,
                "Error: memory allocation failed.\n"
            );

            for (int j = 0; j < i; j++) {
                free(thread_data[j].primes);
            }

            free(threads);
            free(thread_data);

            return 1;
        }

        thread_data[i].count = 0;
    }


    /*
     * Divide the range [2, n) between the threads.
     *
     * We use block partitioning:
     *
     * Thread 0 -> first block
     * Thread 1 -> second block
     * Thread 2 -> third block
     * ...
     */
    int total_numbers = n - 2;

    int chunk_size =
        total_numbers / num_threads;

    int remainder =
        total_numbers % num_threads;


    /*
     * Start measuring wall-clock time.
     */
    struct timespec start_time;
    struct timespec end_time;

    clock_gettime(
        CLOCK_MONOTONIC,
        &start_time
    );


    /*
     * Create the threads.
     */
    int current = 2;

    for (int i = 0; i < num_threads; i++) {

        /*
         * Give the first 'remainder' threads
         * one additional number.
         */
        int thread_size = chunk_size;

        if (i < remainder) {
            thread_size++;
        }


        thread_data[i].start = current;
        thread_data[i].end = current + thread_size;


        pthread_create(
            &threads[i],
            NULL,
            find_primes,
            &thread_data[i]
        );


        current += thread_size;
    }


    /*
     * Wait for every thread to finish.
     */
    for (int i = 0; i < num_threads; i++) {

        pthread_join(
            threads[i],
            NULL
        );
    }


    /*
     * Stop measuring time.
     */
    clock_gettime(
        CLOCK_MONOTONIC,
        &end_time
    );


    /*
     * Calculate elapsed wall-clock time.
     */
    double elapsed_seconds =
        (end_time.tv_sec - start_time.tv_sec) +
        (end_time.tv_nsec - start_time.tv_nsec)
        / 1e9;


    /*
     * Allocate the final array containing all primes.
     */
    int *primes =
        malloc(
            (size_t) n *
            sizeof(int)
        );


    if (primes == NULL) {

        fprintf(
            stderr,
            "Error: memory allocation failed.\n"
        );

        for (int i = 0; i < num_threads; i++) {
            free(thread_data[i].primes);
        }

        free(threads);
        free(thread_data);

        return 1;
    }


    /*
     * Add 2 first, exactly as in Task 1.
     */
    int count = 0;

    primes[count++] = 2;


    /*
     * Combine each thread's results.
     *
     * Since each thread received a consecutive range,
     * and the threads are combined in range order,
     * the final result remains sorted.
     */
    for (int i = 0; i < num_threads; i++) {

        for (int j = 0;
             j < thread_data[i].count;
             j++) {

            primes[count++] =
                thread_data[i].primes[j];
        }
    }


    /*
     * Output results.
     */
    if (n < 100) {

        for (int i = 0; i < count; i++) {
            printf("%d ", primes[i]);
        }

        printf("\n");

    } else {

        FILE *fp =
            fopen("output.txt", "w");


        if (fp == NULL) {

            fprintf(
                stderr,
                "Error: could not open output.txt.\n"
            );

            free(primes);

            for (int i = 0; i < num_threads; i++) {
                free(thread_data[i].primes);
            }

            free(threads);
            free(thread_data);

            return 1;
        }


        for (int i = 0; i < count; i++) {

            fprintf(
                fp,
                "%d\n",
                primes[i]
            );
        }


        fclose(fp);


        printf(
            "Found %d prime numbers less than %d.\n",
            count,
            n
        );

        printf(
            "Output written to output.txt\n"
        );
    }


    /*
     * Print benchmark information.
     */
    printf(
        "Number of threads: %d\n",
        num_threads
    );

    printf(
        "Time taken: %.6f seconds\n",
        elapsed_seconds
    );


    /*
     * Free allocated memory.
     */
    free(primes);

    for (int i = 0; i < num_threads; i++) {
        free(thread_data[i].primes);
    }

    free(threads);
    free(thread_data);


    return 0;
}