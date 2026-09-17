/*
 * FIT3143 Lab 2 - Task 1: Prime search with Open MPI
 *
 * Finds all primes less than n. The root writes them in ascending order
 * to output_mpi.txt (or prints them if n < 100).
 *
 * Only odd numbers are tested. Candidate j is the number k = 3 + 2j,
 * for j = 0 .. num_candidates-1. Each rank tests its share of candidates.
 *
 * Schemes (how candidates are split between ranks):
 *   0 BLOCK    - equal contiguous ranges. Unbalanced, because larger
 *                numbers take longer to test.
 *   1 CYCLIC   - chunks of 1000 candidates dealt round-robin. Balanced,
 *                but the root has to put the results back in order.
 *   2 WEIGHTED - contiguous ranges sized so each rank gets roughly the
 *                same amount of work (boundary at n * (i/p)^(2/3)).
 *
 * Timing starts after buffers are allocated and ends after the file is
 * written. The reported time is the slowest rank's time.
 *
 * Build: mpicc task1.c -o task1_mpi -O2 -lm
 * Run:   srun ./task1_mpi <n> [scheme 0|1|2] [label]
 *        label (optional) replaces the scheme name in the CSV line.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define OUTPUT_FILE "output_mpi.txt"

/* Chunk size for the cyclic scheme (same as the pthreads and OpenMP versions) */
#define CHUNK 1000

#define SCHEME_BLOCK    0
#define SCHEME_CYCLIC   1
#define SCHEME_WEIGHTED 2

static const char *scheme_name(int s) {
    if (s == SCHEME_BLOCK)    return "block";
    if (s == SCHEME_CYCLIC)   return "cyclic";
    if (s == SCHEME_WEIGHTED) return "weighted";
    return "unknown";
}

/* Upper bound on the number of primes below x (Rosser-Schoenfeld).
 * Used to size result buffers. */
static long prime_count_bound(long x) {
    if (x < 100) {
        return x;
    }
    return (long) (1.25506 * (double) x / log((double) x)) + 1;
}

/* Trial division up to sqrt(k). Same function as the serial version. */
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

/* Start index of rank i's range for BLOCK and WEIGHTED.
 * Rank i owns candidates [boundary(i), boundary(i+1)). */
static long range_boundary(int i, int p, long num_candidates, int n, int scheme) {

    if (i <= 0) {
        return 0;
    }
    if (i >= p) {
        return num_candidates;
    }

    if (scheme == SCHEME_WEIGHTED) {
        /* Boundary value k_i = n * (i/p)^(2/3), converted to an index */
        double frac = (double) i / (double) p;
        double k_i  = (double) n * pow(frac, 2.0 / 3.0);
        long   j    = (long) ((k_i - 3.0) / 2.0);

        if (j < 0)               j = 0;
        if (j > num_candidates)  j = num_candidates;
        return j;
    }

    return num_candidates * i / p;
}

/* Writes one prime per line. Builds the whole file in memory and writes it
 * with a single fwrite, which is faster than calling fprintf for every
 * prime. Returns 0 on success. */
static int write_primes(const char *path, const int *primes, int count) {

    /* Up to 10 digits per number, plus a newline */
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

/* CYCLIC only: put the gathered results back in ascending order.
 *
 * Each rank's list is sorted, but the lists overlap. Chunk c belongs to
 * rank c % p and holds the numbers below 3 + 2*(c+1)*CHUNK, so we go
 * through the chunks in order and copy that chunk's primes from its
 * owner's list. Each prime is copied once, so this is O(total). */
static void merge_runs(const int *segments, const int *counts, const int *displs,
                       int p, int total, long num_candidates, int *out) {

    int *pos = calloc((size_t) p, sizeof(int));   /* read position in each list */
    if (pos == NULL) {
        fprintf(stderr, "Root: merge allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long total_chunks = (num_candidates + CHUNK - 1) / CHUNK;
    if (total_chunks < 1) {
        total_chunks = 1;   /* so the prime 2 is still copied when n = 3 */
    }

    int written = 0;

    for (long c = 0; c < total_chunks; c++) {

        int         r     = (int) (c % p);
        long        limit = 3 + 2 * (c + 1) * (long) CHUNK;
        const int  *run   = segments + displs[r];

        while (pos[r] < counts[r] && run[pos[r]] < limit) {
            out[written++] = run[pos[r]++];
        }
    }

    free(pos);

    if (written != total) {
        fprintf(stderr, "Root: merge placed %d of %d primes.\n", written, total);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Read arguments on rank 0 and broadcast them ---- */
    /* params = {n, scheme}. n = 0 tells every rank to stop (bad input). */
    int params[2] = {0, SCHEME_BLOCK};

    if (rank == 0) {
        if (argc < 2 || argc > 4) {
            fprintf(stderr,
                    "Usage: %s <n> [scheme 0=block 1=cyclic 2=weighted] [label]\n",
                    argv[0]);
        } else {
            params[0] = atoi(argv[1]);
            if (argc >= 3) {
                params[1] = atoi(argv[2]);
            }
            if (params[0] <= 2) {
                fprintf(stderr, "n must be greater than 2.\n");
                params[0] = 0;
            }
            if (params[1] < 0 || params[1] > 2) {
                fprintf(stderr, "scheme must be 0, 1 or 2.\n");
                params[0] = 0;
            }
        }
    }

    double t_bcast0 = MPI_Wtime();
    MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD);
    double t_bcast = MPI_Wtime() - t_bcast0;

    int n      = params[0];
    int scheme = params[1];

    const char *label = NULL;   /* only rank 0 prints, so no need to broadcast */
    if (rank == 0 && argc == 4) {
        label = argv[3];
    }

    if (n == 0) {
        MPI_Finalize();
        return 1;
    }

    long num_candidates = ((long) n - 2) / 2;

    /* ---- Allocate this rank's result buffer ---- */
    long jlo = 0;
    long jhi = 0;
    long my_candidates;

    if (scheme == SCHEME_CYCLIC) {
        my_candidates = num_candidates / size + 2 * CHUNK;   /* upper bound */
    } else {
        jlo = range_boundary(rank,     size, num_candidates, n, scheme);
        jhi = range_boundary(rank + 1, size, num_candidates, n, scheme);
        my_candidates = jhi - jlo;
    }

    /* A rank can't find more primes than it has candidates, or more than
     * the prime bound. +2 leaves room for the prime 2 on rank 0. */
    long rs_bound = prime_count_bound((long) n);
    long capacity = (my_candidates < rs_bound ? my_candidates : rs_bound) + 2;

    int *local_primes = malloc((size_t) capacity * sizeof(int));
    if (local_primes == NULL) {
        fprintf(stderr, "Rank %d: local allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *counts = NULL;   /* primes found by each rank (root only) */
    int *displs = NULL;   /* where each rank's primes go (root only) */

    if (rank == 0) {
        counts = malloc((size_t) size * sizeof(int));
        displs = malloc((size_t) size * sizeof(int));
        if (counts == NULL || displs == NULL) {
            fprintf(stderr, "Rank 0: metadata allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* ---- Start timing (barrier so all ranks start together) ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    int local_count = 0;

    /* 2 is the only even prime, so rank 0 adds it directly */
    if (rank == 0) {
        local_primes[local_count++] = 2;
    }

    /* ---- Search this rank's candidates ---- */
    if (scheme == SCHEME_CYCLIC) {

        /* Rank r takes chunks r, r+size, r+2*size, ... */
        for (long c = rank; c * CHUNK < num_candidates; c += size) {

            long chunk_lo = c * CHUNK;
            long chunk_hi = chunk_lo + CHUNK;
            if (chunk_hi > num_candidates) {
                chunk_hi = num_candidates;
            }

            for (long j = chunk_lo; j < chunk_hi; j++) {
                int k = 3 + 2 * (int) j;
                if (is_prime(k)) {
                    local_primes[local_count++] = k;
                }
            }
        }

    } else {

        for (long j = jlo; j < jhi; j++) {
            int k = 3 + 2 * (int) j;
            if (is_prime(k)) {
                local_primes[local_count++] = k;
            }
        }
    }

    double t_search_end = MPI_Wtime();

    /* ---- Collect results on the root ---- */
    /* Step 1: gather how many primes each rank found */
    double t_comm0 = MPI_Wtime();

    MPI_Gather(&local_count, 1, MPI_INT,
               counts,       1, MPI_INT,
               0, MPI_COMM_WORLD);

    double t_gather = MPI_Wtime() - t_comm0;

    int    total      = 0;
    int   *all_primes = NULL;
    double t_prefix   = 0.0;

    /* Step 2: prefix sum of the counts gives each rank's offset */
    if (rank == 0) {
        double t_p0 = MPI_Wtime();
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + counts[i - 1];
        }
        total = displs[size - 1] + counts[size - 1];
        t_prefix = MPI_Wtime() - t_p0;

        all_primes = malloc((size_t) total * sizeof(int));
        if (all_primes == NULL) {
            fprintf(stderr, "Rank 0: receive buffer allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Step 3: gather the primes themselves */
    double t_gv0 = MPI_Wtime();

    MPI_Gatherv(local_primes, local_count, MPI_INT,
                all_primes, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    double t_comm = t_gather + (MPI_Wtime() - t_gv0);

    /* ---- Sort order ---- */
    /* Block and weighted results are already in order.
     * Cyclic results need to be merged. */
    double t_merge_start = MPI_Wtime();

    if (rank == 0 && scheme == SCHEME_CYCLIC) {
        int *merged = malloc((size_t) total * sizeof(int));
        if (merged == NULL) {
            fprintf(stderr, "Rank 0: merge buffer allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        merge_runs(all_primes, counts, displs, size, total,
                   num_candidates, merged);
        free(all_primes);
        all_primes = merged;
    }

    double t_end = MPI_Wtime();

    /* ---- Timing results ---- */
    double my_total  = (t_end - t_start) + t_bcast;
    double my_search = t_search_end - t_start;
    double my_merge  = t_end - t_merge_start;
    double my_comm   = t_comm;

    double max_total  = 0.0;
    double max_search = 0.0;
    double min_search = 0.0;
    double max_comm   = 0.0;

    /* The program is only done when the slowest rank is done, so use the max */
    MPI_Reduce(&my_total,  &max_total,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_search, &max_search, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_search, &min_search, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_comm,   &max_comm,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ---- Count how many nodes were used ---- */
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int  name_len = 0;
    MPI_Get_processor_name(hostname, &name_len);

    char *all_names = NULL;
    if (rank == 0) {
        all_names = malloc((size_t) size * MPI_MAX_PROCESSOR_NAME);
        if (all_names == NULL) {
            fprintf(stderr, "Rank 0: hostname buffer allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(hostname,  MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               all_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               0, MPI_COMM_WORLD);

    /* ---- Output and report (root only) ---- */
    if (rank == 0) {

        int nodes = 0;   /* number of distinct hostnames */
        for (int i = 0; i < size; i++) {
            char *name_i = all_names + (size_t) i * MPI_MAX_PROCESSOR_NAME;
            int seen = 0;
            for (int j = 0; j < i; j++) {
                if (strcmp(name_i, all_names + (size_t) j * MPI_MAX_PROCESSOR_NAME) == 0) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) {
                nodes++;
            }
        }

        /* Write the output (timed, added to the total) */
        double t_write0 = MPI_Wtime();

        if (n < 100) {
            for (int i = 0; i < total; i++) {
                printf("%d ", all_primes[i]);
            }
            printf("\n");
        } else if (write_primes(OUTPUT_FILE, all_primes, total) != 0) {
            fprintf(stderr, "Error: could not write %s.\n", OUTPUT_FILE);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double t_write = MPI_Wtime() - t_write0;

        /* Imbalance: gap between slowest and fastest rank's search time */
        double imbalance = 0.0;
        if (max_search > 0.0) {
            imbalance = 100.0 * (max_search - min_search) / max_search;
        }

        /* Split the total time for Amdahl's Law:
         *   parallel = search (slowest rank)
         *   serial   = broadcast + prefix sum + merge + file write
         *   overhead = everything else (mostly communication) */
        double end_to_end    = max_total + t_write;
        double serial_part   = t_bcast + t_prefix + my_merge + t_write;
        double parallel_part = max_search;
        double overhead      = end_to_end - serial_part - parallel_part;
        if (overhead < 0.0) {
            overhead = 0.0;
        }

        printf("scheme=%s n=%d procs=%d nodes=%d primes=%d\n",
               scheme_name(scheme), n, size, nodes, total);
        printf("  total time      : %.6f s  (incl. file write)\n", end_to_end);
        printf("  parallel (search): %.6f s\n", parallel_part);
        printf("  serial parts     : %.6f s\n", serial_part);
        printf("    broadcast      : %.6f s\n", t_bcast);
        printf("    prefix sum     : %.6f s\n", t_prefix);
        printf("    merge          : %.6f s\n", my_merge);
        printf("    file write     : %.6f s\n", t_write);
        printf("  overhead (comm)  : %.6f s\n", overhead);
        printf("    collectives    : %.6f s\n", max_comm);
        printf("  search (fastest) : %.6f s\n", min_search);
        printf("  imbalance        : %.2f %%\n", imbalance);

        /* CSV line, same columns as the other versions:
         * impl,scheme,n,procs,threads,nodes,primes,
         * total,serial,parallel,overhead,imbalance,write */
        printf("CSV,mpi,%s,%d,%d,1,%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.6f\n",
               label ? label : scheme_name(scheme), n, size, nodes, total,
               end_to_end, serial_part, parallel_part, overhead, imbalance,
               t_write);

        free(all_names);
        free(all_primes);
        free(counts);
        free(displs);
    }

    free(local_primes);

    MPI_Finalize();
    return 0;
}