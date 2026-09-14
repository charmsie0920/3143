/*
 * Task 1 - Open MPI: Finding Prime Numbers
 * FIT3143 Lab 2 (Week 8)
 *
 * Parallel prime search using MPI. Finds all primes strictly less than n
 * and writes them, in ascending order, to a text file from the root process.
 *
 * This version adds the three workload distribution schemes and the timing
 * instrumentation. The prime test is unchanged from the serial, pthreads
 * and OpenMP versions, so any difference in runtime is parallelisation and
 * not a different algorithm.
 *
 * ---- The three distribution schemes ----------------------------------------
 *
 * The candidates are the odd numbers 3, 5, 7, ... below n, treated as an
 * index space 0 .. num_candidates-1 where candidate j is the number
 * k = 3 + 2j. The schemes differ only in which candidates a rank owns.
 *
 *   0  BLOCK -- equal-sized contiguous ranges.
 *      Simple, and output is sorted for free. But cost per candidate grows
 *      with k, so the last rank does far more work than the first. Expect
 *      poor balance and poor speedup: this is the baseline to beat.
 *
 *   1  CYCLIC -- fixed-size chunks handed out round-robin.
 *      Rank r takes chunks r, r+p, r+2p, ... Every rank gets a fair mix of
 *      cheap and expensive candidates, so the compute balance is good.
 *      The cost is that each rank's primes are interleaved with everyone
 *      else's, so the concatenation Gatherv produces is NOT sorted and the
 *      root must merge p sorted runs afterwards. That merge is serial work
 *      on the root and is measured separately below.
 *
 *   2  WEIGHTED -- contiguous ranges sized by estimated cost.
 *      Trial division to sqrt(k) costs about sqrt(k) work per candidate, so
 *      cumulative work up to x grows roughly as x^1.5. Putting the boundary
 *      for rank i at k_i = n * (i/p)^(2/3) therefore gives every rank about
 *      the same amount of WORK rather than the same COUNT of candidates.
 *      Ranges stay contiguous and ascending, so output is still sorted for
 *      free -- balance without paying for a merge.
 *
 *
 * ---- Timing ----------------------------------------------------------------
 *
 * The timed region matches the revised Week 4 versions exactly: it starts
 * after buffers are allocated and ends when the sorted list has been written
 * to the output file. Writing the file is serial work on the root, so it is
 * also reported on its own and counted in Amdahl's serial fraction.
 *
 * Parallel runtime is the MAXIMUM over ranks, not rank 0's own time -- the
 * job is not finished until the slowest rank is finished. MPI_Reduce with
 * MPI_MAX does this. An MPI_Barrier before the start clock stops ranks that
 * happened to reach the region early from recording an unfairly long time.
 *
 * Known limitation: MPI_Gatherv counts and displacements are int, so the
 * total prime count must fit in an int. Fine to around n = 10^9.
 *
 * Build: module load openmpi/4.1.5-gcc-11.2.0-ux65npg
 *        mpicc task1_mpi.c -o task1_mpi -O2 -lm
 * Run:   srun ./task1_mpi <n> [scheme 0|1|2] [label]   (inside a SLURM job)
 *
 * The optional label replaces the scheme name in the CSV output. It exists so
 * weak-scaling runs can be tagged (e.g. "cyclic-weak") and kept separate from
 * strong-scaling runs during analysis -- otherwise a weak-scaling point that
 * happens to share an n value with a strong-scaling point gets averaged in
 * with it and both become meaningless.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>

#define OUTPUT_FILE "output_mpi.txt"

/* Candidates per chunk for the cyclic scheme. Matches the pthreads CHUNK
 * and the OpenMP schedule(dynamic, 1000) so all three parallel versions
 * use the same granularity. */
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

/* Rosser & Schoenfeld (1962): pi(x) < 1.25506 * x / ln(x) for x > 1. */
static long prime_count_bound(long x) {
    if (x < 100) {
        return x;
    }
    return (long) (1.25506 * (double) x / log((double) x)) + 1;
}

/* Returns 1 if k is prime, 0 otherwise. Identical across all four versions. */
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

/* Boundary of rank i's range in candidate-index space, for the contiguous
 * schemes. Rank i owns [boundary(i), boundary(i+1)).
 *
 * BLOCK: equal candidate counts. Multiplying before dividing spreads the
 * remainder so blocks differ by at most one and nothing is dropped.
 *
 * WEIGHTED: equal estimated cost. k_i = n * (i/p)^(2/3), converted to a
 * candidate index by j = (k - 3) / 2. The endpoints are pinned exactly so
 * that rounding can never lose or duplicate a candidate at the ends. */
static long range_boundary(int i, int p, long num_candidates, int n, int scheme) {

    if (i <= 0) {
        return 0;
    }
    if (i >= p) {
        return num_candidates;
    }

    if (scheme == SCHEME_WEIGHTED) {
        double frac = (double) i / (double) p;
        double k_i  = (double) n * pow(frac, 2.0 / 3.0);
        long   j    = (long) ((k_i - 3.0) / 2.0);

        if (j < 0)               j = 0;
        if (j > num_candidates)  j = num_candidates;
        return j;
    }

    return num_candidates * i / p;
}

/* Write primes one per line to path. fprintf re-parses its format string for
 * every number; formatting the digits by hand into one buffer and issuing a
 * single fwrite gives byte-identical output without that per-number cost.
 * The same writer is used in every version, so the file write costs the
 * same everywhere and the speedups stay comparable. Returns 0 on success. */
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

/* Merge p ascending runs into one ascending array.
 * segments[displs[r] .. displs[r]+counts[r]) is run r. Simple linear scan
 * over the p run heads: O(total * p). With p at most 32 that is cheaper in
 * practice than a heap and far simpler to read. Only needed for CYCLIC. */
static void merge_runs(const int *segments, const int *counts, const int *displs,
                       int p, int total, int *out) {

    int *pos = calloc((size_t) p, sizeof(int));
    if (pos == NULL) {
        fprintf(stderr, "Root: merge allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int written = 0; written < total; written++) {

        int best_rank  = -1;
        int best_value = INT_MAX;

        for (int r = 0; r < p; r++) {
            if (pos[r] < counts[r]) {
                int v = segments[displs[r] + pos[r]];
                if (v < best_value) {
                    best_value = v;
                    best_rank  = r;
                }
            }
        }

        out[written] = best_value;
        pos[best_rank]++;
    }

    free(pos);
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Rank 0 reads the arguments, then broadcasts them ---------------
     * n = 0 is the "bad input, everyone stop" signal. Without it one rank
     * would exit while the others blocked forever in the broadcast. */
    int params[2] = {0, SCHEME_BLOCK};

    if (rank == 0) {
        if (argc < 2 || argc > 4) {
            fprintf(stderr,
                    "Usage: %s <n> [scheme 0=block 1=cyclic 2=weighted] [label]\n",
                    argv[0]);
        } else {
            params[0] = atoi(argv[1]);
            if (argc == 3) {
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

    /* The broadcast is serial work in Amdahl's sense: its cost does not fall
     * as ranks are added. Timed so it can be counted in the serial fraction. */
    double t_bcast0 = MPI_Wtime();
    MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD);
    double t_bcast = MPI_Wtime() - t_bcast0;

    int n      = params[0];
    int scheme = params[1];

    /* Only the root prints, so the label needs no broadcast. */
    const char *label = NULL;
    if (rank == 0 && argc == 4) {
        label = argv[3];
    }

    if (n == 0) {
        MPI_Finalize();
        return 1;
    }

    long num_candidates = ((long) n - 2) / 2;

    /* ---- Local buffer, sized before the clock starts --------------------
     * A rank cannot find more primes than it has candidates, so its own
     * candidate count is a safe capacity; the Rosser-Schoenfeld bound is
     * tighter for large blocks, so take the smaller. +2 covers rank 0's
     * extra entry for the prime 2 and leaves slack for empty ranges. */
    long jlo = 0;
    long jhi = 0;
    long my_candidates;

    if (scheme == SCHEME_CYCLIC) {
        /* Rank r owns chunks r, r+p, ... Bounding the count is enough here;
         * an exact count would need a pass over the chunk indices. */
        my_candidates = num_candidates / size + 2 * CHUNK;
    } else {
        jlo = range_boundary(rank,     size, num_candidates, n, scheme);
        jhi = range_boundary(rank + 1, size, num_candidates, n, scheme);
        my_candidates = jhi - jlo;
    }

    long rs_bound = prime_count_bound((long) n);
    long capacity = (my_candidates < rs_bound ? my_candidates : rs_bound) + 2;

    int *local_primes = malloc((size_t) capacity * sizeof(int));
    if (local_primes == NULL) {
        fprintf(stderr, "Rank %d: local allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *counts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        counts = malloc((size_t) size * sizeof(int));
        displs = malloc((size_t) size * sizeof(int));
        if (counts == NULL || displs == NULL) {
            fprintf(stderr, "Rank 0: metadata allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* ---- Timed region begins --------------------------------------------
     * The barrier means every rank starts the clock at the same moment, so
     * a rank that arrived early does not report time it spent waiting. */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    int local_count = 0;

    /* 2 is the only even prime and is never generated by k = 3 + 2j. Rank 0
     * contributes it directly, and it belongs at the front of the list. */
    if (rank == 0) {
        local_primes[local_count++] = 2;
    }

    if (scheme == SCHEME_CYCLIC) {

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

    /* ---- Collection on the root -----------------------------------------
     * Gather the per-rank counts, prefix-sum them into displacements, then
     * Gatherv the data. displs[i] is where rank i's block lands, which is
     * the same prefix-sum idea used by the threaded compaction, applied
     * across processes instead of threads. */
    double t_comm0 = MPI_Wtime();

    MPI_Gather(&local_count, 1, MPI_INT,
               counts,       1, MPI_INT,
               0, MPI_COMM_WORLD);

    double t_gather = MPI_Wtime() - t_comm0;

    int    total      = 0;
    int   *all_primes = NULL;
    double t_prefix   = 0.0;

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

    double t_gv0 = MPI_Wtime();

    MPI_Gatherv(local_primes, local_count, MPI_INT,
                all_primes, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Communication only: the prefix sum in between is counted as serial. */
    double t_comm = t_gather + (MPI_Wtime() - t_gv0);

    /* ---- Merge, only for cyclic ------------------------------------------
     * Block and weighted produce ascending ranges in ascending rank order,
     * so the concatenation is already sorted. Cyclic interleaves, so the
     * root has to merge. This is the price cyclic pays for its balance. */
    double t_merge_start = MPI_Wtime();

    if (rank == 0 && scheme == SCHEME_CYCLIC) {
        int *merged = malloc((size_t) total * sizeof(int));
        if (merged == NULL) {
            fprintf(stderr, "Rank 0: merge buffer allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        merge_runs(all_primes, counts, displs, size, total, merged);
        free(all_primes);
        all_primes = merged;
    }

    double t_end = MPI_Wtime();
    /* ---- Timed region ends: sorted list now exists on the root ---------- */

    double my_total  = (t_end - t_start) + t_bcast;
    double my_search = t_search_end - t_start;
    double my_merge  = t_end - t_merge_start;
    double my_comm   = t_comm;

    double max_total  = 0.0;
    double max_search = 0.0;
    double min_search = 0.0;
    double max_comm   = 0.0;

    /* Parallel runtime is the slowest rank, not rank 0. */
    MPI_Reduce(&my_total,  &max_total,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_search, &max_search, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_search, &min_search, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_comm,   &max_comm,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ---- How many distinct nodes were actually used ---------------------
     * Recorded so the CSV distinguishes a single-node run from a run spread
     * across the gigabit network at the same process count. */
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

    /* ---- Output and reporting (root only) -------------------------------- */
    if (rank == 0) {

        int nodes = 0;
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

        /* ---- File output -------------------------------------------------
         * Serial work on the root, and part of what the user waits for, so
         * it is timed and added to the total below. */
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

        /* Imbalance: 0% means every rank finished searching together, which
         * is the ideal. A large value means ranks sat idle waiting. */
        double imbalance = 0.0;
        if (max_search > 0.0) {
            imbalance = 100.0 * (max_search - min_search) / max_search;
        }

        /* ---- Phase accounting for Amdahl's Law ---------------------------
         * parallel : the search, which divides by the process count
         * serial   : broadcast + prefix sum + merge + file write -- work whose
         *            cost does NOT fall as ranks are added
         * overhead : the collective communication, which actually GROWS with
         *            the process count and is the term Amdahl does not model
         *
         * Anything unaccounted for (barrier waits, scheduling jitter) is
         * folded into overhead so the three always sum to the total. */
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

        /* Unified CSV, same column layout as the serial, pthreads and OpenMP
         * versions so one parser handles every result file.
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