/*
 * Task 1 - Serial Code: Finding Prime Numbers
 * FIT3143 Lab 1 (Week 4)   [REVISED for Lab 2 / Week 8]
 *
 * Finds all prime numbers strictly less than a user-provided integer n.
 *
 * ---- Revisions for Lab 2 -----------------------------------------------------
 * 1. Timing changed from clock() (CPU time) to clock_gettime(CLOCK_MONOTONIC)
 *    (wall-clock), matching Tasks 2, 3 and the MPI versions.
 * 2. Timed region defined identically across ALL versions (serial, pthreads,
 *    OpenMP, MPI, hybrid): from the start of the search until the sorted
 *    prime list has been output (console if n < 100, otherwise output.txt).
 *    Buffer allocation is excluded. The file write uses the same
 *    write_primes() everywhere.
 * 3. Result buffer sized by the Rosser-Schoenfeld bound rather than n ints.
 * 4. Emits a machine-readable CSV line so one parser handles every version.
 *
 * ---- Serial and parallel parts (Amdahl's Law) ------------------------------
 * The search loop is the parallelisable part: every iteration is independent.
 * The output (file write) is the serial part: it happens once, on one process, in
 * every version. So parallel_s = search time and serial_s = write time, and
 * f = serial_s / total_s is measured directly from this program.
 *
 * Compile: gcc task1.c -o task1 -O2 -lm
 * Run:     ./task1 <n>
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

/* Rosser & Schoenfeld (1962): pi(x) < 1.25506 * x / ln(x) for x > 1. */
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

    long capacity = prime_count_bound((long) n);
    int *primes = malloc((size_t) capacity * sizeof(int));
    if (primes == NULL) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        return 1;
    }

    /* ---- Timed region --------------------------------------------------- */
    double start_time = now_seconds();

    int count = 0;
    primes[count++] = 2;
    for (int k = 3; k < n; k += 2) {
        if (is_prime(k)) {
            primes[count++] = k;
        }
    }

    double elapsed_seconds = now_seconds() - start_time;
    /* ---- Search done; the file write below is also timed ---------------- */

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
        return 1;
    }

    double t_write = now_seconds() - t_write0;
    double total_seconds = elapsed_seconds + t_write;

    printf("Primes found: %d\n", count);
    printf("Total time (incl. file write): %.6f seconds\n", total_seconds);
    printf("  parallelisable (search) : %.6f s\n", elapsed_seconds);
    printf("  serial (file write)     : %.6f s\n", t_write);

    /* impl,scheme,n,procs,threads,nodes,primes,
     * total,serial,parallel,overhead,imbalance,write */
    printf("CSV,serial,none,%d,1,1,1,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.6f\n",
           n, count, total_seconds, t_write, elapsed_seconds, 0.0, 0.0,
           t_write);

    free(primes);
    return 0;
}