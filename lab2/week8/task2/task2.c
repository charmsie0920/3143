/*
 * FIT3143 Lab 2 - Task 2: Prime search with Open MPI + OpenMP
 *
 * Finds all primes less than n. The root writes them in ascending order
 * to output_hybrid.txt (or prints them if n < 100).
 *
 * Only odd numbers are tested. Candidate j is the number k = 3 + 2j.
 *
 * Two levels of parallelism:
 *   Level 1 - candidates are split between MPI ranks (same as Task 1).
 *   Level 2 - each rank splits its share between its OpenMP threads,
 *             using the same scheme.
 *
 * Schemes:
 *   0 BLOCK    - equal contiguous ranges. Unbalanced.
 *   1 CYCLIC   - chunks of 1000 candidates dealt round-robin to ranks.
 *                Within a rank, threads take chunks dynamically.
 *   2 WEIGHTED - contiguous ranges sized for roughly equal work.
 *
 * Avoiding races between threads:
 *   A rank's share is cut into "slots" (one per thread for block/weighted,
 *   one per chunk for cyclic). Each slot writes to its own part of a
 *   scratch array and keeps its own count. Afterwards a prefix sum of the
 *   counts gives each slot's position in the final array, and the slots
 *   are copied there. Slots are in ascending order, so the result is
 *   sorted without a sort.
 *
 * Timing starts after buffers are allocated and ends after the file is
 * written. The reported time is the slowest rank's time.
 *
 * Build: mpicc task2.c -o task2_hybrid -O2 -fopenmp -lm
 * Run:   srun ./task2_hybrid <n> <threads> [scheme 0|1|2] [label]
 *        Allocate with --ntasks=<procs> --cpus-per-task=<threads>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define OUTPUT_FILE "output_hybrid.txt"

/* Chunk size for the cyclic scheme (same as Task 1) */
#define CHUNK 1000

/* Sanity limit on the thread count argument */
#define MAX_THREADS 256

#define SCHEME_BLOCK    0
#define SCHEME_CYCLIC   1
#define SCHEME_WEIGHTED 2

static const char *scheme_name(int s) {
    if (s == SCHEME_BLOCK)    return "block";
    if (s == SCHEME_CYCLIC)   return "cyclic";
    if (s == SCHEME_WEIGHTED) return "weighted";
    return "unknown";
}

/* Upper bound on the number of primes below x (Rosser-Schoenfeld) */
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

/* Level 1: start index of rank i's range for BLOCK and WEIGHTED.
 * Rank i owns candidates [boundary(i), boundary(i+1)). Same as Task 1. */
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

/* Level 2: start index of thread t's part of a rank's range [jlo, jhi).
 * Same idea as range_boundary, but for a range that doesn't start at 0.
 * For WEIGHTED, work up to k grows like k^1.5, so the boundary is
 *   k_t = (klo^1.5 + (t/nt) * (khi^1.5 - klo^1.5))^(2/3) */
static long split_boundary(int t, int nt, long jlo, long jhi, int scheme) {

    if (t <= 0) {
        return jlo;
    }
    if (t >= nt) {
        return jhi;
    }

    if (scheme == SCHEME_WEIGHTED) {
        double klo  = 3.0 + 2.0 * (double) jlo;
        double khi  = 3.0 + 2.0 * (double) jhi;
        double lo15 = pow(klo, 1.5);
        double hi15 = pow(khi, 1.5);
        double frac = (double) t / (double) nt;
        double k_t  = pow(lo15 + frac * (hi15 - lo15), 2.0 / 3.0);
        long   j    = (long) ((k_t - 3.0) / 2.0);

        if (j < jlo) j = jlo;
        if (j > jhi) j = jhi;
        return j;
    }

    return jlo + (jhi - jlo) * t / nt;
}

/* Upper bound on the number of primes in [lo, hi), using
 * pi(hi) < 1.25506 hi/ln hi and pi(lo) > lo/ln lo (valid for lo >= 17). */
static long prime_count_bound_range(long lo, long hi) {

    if (hi <= lo) {
        return 0;
    }
    if (lo < 17) {
        return prime_count_bound(hi) + 2;
    }

    double upper = 1.25506 * (double) hi / log((double) hi);
    double lower = (double) lo / log((double) lo);
    double diff  = upper - lower;

    if (diff < 0.0) {
        diff = 0.0;
    }
    return (long) diff + 2;
}

/* How much scratch space a slot needs: the smallest of
 *   - the number of candidates in the slot
 *   - the Rosser-Schoenfeld range bound (good for wide slots)
 *   - Montgomery-Vaughan: pi(x+y) - pi(x) <= 2y / ln y (good for narrow
 *     slots such as a 1000-candidate cyclic chunk) */
static long slot_capacity(long lo, long hi) {

    long cand = hi - lo;
    if (cand <= 0) {
        return 0;
    }

    long cap = cand;

    long rs = prime_count_bound_range(3 + 2 * lo, 3 + 2 * hi);
    if (rs < cap) {
        cap = rs;
    }

    double y  = 2.0 * (double) cand;   /* the slot spans 2 * cand integers */
    long   mv = (long) (2.0 * y / log(y)) + 2;
    if (mv < cap) {
        cap = mv;
    }

    return cap;
}

/* Writes one prime per line. Builds the whole file in memory and writes it
 * with a single fwrite, which is faster than calling fprintf for every
 * prime. Returns 0 on success. Same as Task 1. */
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
 * owner's list. Each prime is copied once, so this is O(total).
 * Same as Task 1. */
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

    /* FUNNELED: threads are used, but only the main thread calls MPI */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI library lacks MPI_THREAD_FUNNELED support "
                        "(requested %d, got %d).\n",
                MPI_THREAD_FUNNELED, provided);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Read arguments on rank 0 and broadcast them ---- */
    /* params = {n, scheme, threads}. n = 0 tells every rank to stop. */
    int params[3] = {0, SCHEME_BLOCK, 1};

    if (rank == 0) {
        if (argc < 3 || argc > 5) {
            fprintf(stderr,
                    "Usage: %s <n> <threads> "
                    "[scheme 0=block 1=cyclic 2=weighted] [label]\n",
                    argv[0]);
        } else {
            params[0] = atoi(argv[1]);
            params[2] = atoi(argv[2]);
            if (argc >= 4) {
                params[1] = atoi(argv[3]);
            }
            if (params[0] <= 2) {
                fprintf(stderr, "n must be greater than 2.\n");
                params[0] = 0;
            }
            if (params[1] < 0 || params[1] > 2) {
                fprintf(stderr, "scheme must be 0, 1 or 2.\n");
                params[0] = 0;
            }
            if (params[2] < 1 || params[2] > MAX_THREADS) {
                fprintf(stderr, "threads must be between 1 and %d.\n",
                        MAX_THREADS);
                params[0] = 0;
            }
        }
    }

    /* All ranks must agree on n, the scheme and the thread count */
    double t_bcast0 = MPI_Wtime();
    MPI_Bcast(params, 3, MPI_INT, 0, MPI_COMM_WORLD);
    double t_bcast = MPI_Wtime() - t_bcast0;

    int n       = params[0];
    int scheme  = params[1];
    int threads = params[2];

    const char *label = NULL;   /* only rank 0 prints, so no need to broadcast */
    if (rank == 0 && argc == 5) {
        label = argv[4];
    }

    if (n == 0) {
        MPI_Finalize();
        return 1;
    }

    /* Always use exactly the requested number of threads */
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    long num_candidates = ((long) n - 2) / 2;

    /* ---- Level 1: this rank's share ---- */
    /* Block/weighted: the range [jlo, jhi).
     * Cyclic: chunks rank, rank+size, rank+2*size, ... (my_chunks of them) */
    long jlo = 0;
    long jhi = 0;

    long total_chunks = (num_candidates + CHUNK - 1) / CHUNK;
    long my_chunks    = 0;

    if (scheme == SCHEME_CYCLIC) {
        if (total_chunks > rank) {
            my_chunks = (total_chunks - rank + size - 1) / size;
        }
    } else {
        jlo = range_boundary(rank,     size, num_candidates, n, scheme);
        jhi = range_boundary(rank + 1, size, num_candidates, n, scheme);
    }

    /* ---- Level 2: cut the share into slots ---- */
    /* Cyclic: one slot per chunk. Block/weighted: one slot per thread. */
    long nslots = (scheme == SCHEME_CYCLIC) ? my_chunks : threads;

    long   *slot_lo = malloc((size_t) (nslots + 1) * sizeof(long));  /* slot start */
    long   *slot_hi = malloc((size_t) (nslots + 1) * sizeof(long));  /* slot end */
    long   *sslice  = malloc((size_t) (nslots + 1) * sizeof(long));  /* scratch offset */
    int    *scount  = calloc((size_t) nslots + 1, sizeof(int));      /* primes per slot */
    double *ttime   = calloc((size_t) threads, sizeof(double));      /* time per thread */

    if (slot_lo == NULL || slot_hi == NULL || sslice == NULL
        || scount == NULL || ttime == NULL) {
        fprintf(stderr, "Rank %d: slot metadata allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Slot s uses scratch[sslice[s] .. sslice[s+1]) */
    sslice[0] = 0;

    for (long s = 0; s < nslots; s++) {

        if (scheme == SCHEME_CYCLIC) {
            slot_lo[s] = (rank + s * size) * CHUNK;
            slot_hi[s] = slot_lo[s] + CHUNK;
            if (slot_hi[s] > num_candidates) {
                slot_hi[s] = num_candidates;
            }
        } else {
            slot_lo[s] = split_boundary((int) s,     threads, jlo, jhi, scheme);
            slot_hi[s] = split_boundary((int) s + 1, threads, jlo, jhi, scheme);
        }

        sslice[s + 1] = sslice[s] + slot_capacity(slot_lo[s], slot_hi[s]);
    }

    /* Cyclic: threads grab chunks one at a time as they finish (dynamic).
     * Block/weighted: thread t gets slot t (static). */
    if (scheme == SCHEME_CYCLIC) {
        omp_set_schedule(omp_sched_dynamic, 1);
    } else {
        omp_set_schedule(omp_sched_static, 1);
    }

    /* +2 leaves room for the prime 2 on rank 0 */
    long capacity = sslice[nslots] + 2;

    int *local_primes = malloc((size_t) capacity * sizeof(int));
    int *scratch      = malloc((size_t) sslice[nslots] * sizeof(int) + 1);

    if (local_primes == NULL || scratch == NULL) {
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

    /* Thread results start after the 2 on rank 0, at 0 elsewhere */
    int base = local_count;

    #pragma omp parallel num_threads(threads)
    {
        /* omp_get_wtime because only the main thread may call MPI */
        double t_thread0 = omp_get_wtime();

        /* ---- Phase A: search the slots ---- */
        /* Each slot has its own scratch space and count, so no locks needed.
         * nowait so each thread records when it finished its own work. */
        #pragma omp for schedule(runtime) nowait
        for (long s = 0; s < nslots; s++) {

            int *mine  = scratch + sslice[s];
            int  found = 0;

            for (long j = slot_lo[s]; j < slot_hi[s]; j++) {
                int k = 3 + 2 * (int) j;
                if (is_prime(k)) {
                    mine[found++] = k;
                }
            }

            scount[s + 1] = found;
        }

        ttime[omp_get_thread_num()] = omp_get_wtime() - t_thread0;
    }

    /* ---- Phase B: prefix sum of the slot counts ---- */
    /* After this, scount[s] = number of primes before slot s */
    for (long s = 0; s < nslots; s++) {
        scount[s + 1] += scount[s];
    }

    /* ---- Phase C: copy each slot's primes into place ---- */
    /* Destinations don't overlap, so this can run in parallel */
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (long s = 0; s < nslots; s++) {
        int found = scount[s + 1] - scount[s];
        if (found > 0) {
            memcpy(local_primes + base + scount[s], scratch + sslice[s],
                   (size_t) found * sizeof(int));
        }
    }

    local_count = base + scount[nslots];

    double t_search_end = MPI_Wtime();

    /* ---- Collect results on the root (same as Task 1) ---- */
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

    /* Thread imbalance on this rank: gap between slowest and fastest thread */
    double my_tmax = ttime[0];
    double my_tmin = ttime[0];

    for (int t = 1; t < threads; t++) {
        if (ttime[t] > my_tmax) my_tmax = ttime[t];
        if (ttime[t] < my_tmin) my_tmin = ttime[t];
    }

    double my_thread_imb = 0.0;
    if (my_tmax > 0.0) {
        my_thread_imb = 100.0 * (my_tmax - my_tmin) / my_tmax;
    }

    double max_total  = 0.0;
    double max_search = 0.0;
    double min_search = 0.0;
    double max_comm   = 0.0;
    double max_thread_imb = 0.0;

    /* The program is only done when the slowest rank is done, so use the max */
    MPI_Reduce(&my_total,  &max_total,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_search, &max_search, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_search, &min_search, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_comm,   &max_comm,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Worst thread imbalance across all ranks */
    MPI_Reduce(&my_thread_imb, &max_thread_imb, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

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

        /* Rank imbalance: gap between slowest and fastest rank's search time */
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

        printf("scheme=%s n=%d procs=%d threads=%d total_workers=%d "
               "nodes=%d primes=%d\n",
               scheme_name(scheme), n, size, threads, size * threads,
               nodes, total);
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
        printf("  imbalance (rank) : %.2f %%\n", imbalance);
        printf("  imbalance (thrd) : %.2f %%\n", max_thread_imb);

        /* CSV line, same columns as the other versions:
         * impl,scheme,n,procs,threads,nodes,primes,
         * total,serial,parallel,overhead,imbalance,write */
        printf("CSV,hybrid,%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.6f\n",
               label ? label : scheme_name(scheme), n, size, threads, nodes,
               total, end_to_end, serial_part, parallel_part, overhead,
               imbalance, t_write);

        free(all_names);
        free(all_primes);
        free(counts);
        free(displs);
    }

    free(local_primes);
    free(scratch);
    free(slot_lo);
    free(slot_hi);
    free(sslice);
    free(scount);
    free(ttime);

    MPI_Finalize();
    return 0;
}