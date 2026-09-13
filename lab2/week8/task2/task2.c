/*
 * Task 2 - Hybrid Open MPI + OpenMP: Finding Prime Numbers
 * FIT3143 Lab 2 (Week 8)
 *
 * Hybrid prime search. Distributed-memory parallelism across MPI processes,
 * shared-memory parallelism across OpenMP threads inside each process. Finds
 * all primes strictly less than n and writes them, in ascending order, to a
 * text file from the root process.
 *
 * Derived from task1_mpi.c. The prime test, the distribution schemes and the
 * timing methodology are all unchanged, so a hybrid run and a pure-MPI run at
 * the same n differ only by the threading layer and are directly comparable.
 *
 * ---- Two levels of partitioning --------------------------------------------
 *
 * The candidates are the odd numbers 3, 5, 7, ... below n, treated as an
 * index space 0 .. num_candidates-1 where candidate j is the number
 * k = 3 + 2j.
 *
 * LEVEL 1 (processes) splits that index space across ranks, using one of the
 * three schemes below. LEVEL 2 (threads) splits each rank's share across its
 * threads using the SAME scheme, one level down. The two levels are the same
 * rule applied recursively, which is why range_boundary() and
 * split_boundary() compute the same thing over different intervals.
 *
 * The schemes differ only in which candidates a rank owns.
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
 *      At the thread level the rank's chunk LIST is split contiguously, not
 *      round-robin a second time. Each thread therefore still produces an
 *      ascending run and the rank's concatenated output stays ascending,
 *      which is what merge_runs() on the root assumes.
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
 * ---- How the threads produce a sorted local array --------------------------
 *
 * task1_mpi.c appends with local_primes[local_count++]. That is a data race
 * the moment more than one thread runs it, and it would also destroy sorted
 * order even with an atomic counter, because threads finish candidates out
 * of order.
 *
 * So each worker is given a slice of a scratch array up front and never
 * touches anyone else's, in three phases:
 *
 *   A  each worker searches its own sub-range into its own scratch slice
 *      and records how many primes it found
 *   B  the per-worker counts are exclusive-prefix-summed, which yields the
 *      offset in local_primes at which each worker's primes belong
 *   C  each worker copies its slice to that offset
 *
 * Sorted order falls out for free: sub-ranges are contiguous and ascending
 * and each worker scans its own in ascending order, so worker i's output is
 * entirely below worker i+1's. This is the same prefix-sum compaction the
 * revised Week 4 pthreads and OpenMP versions use, and the same idea as the
 * Gather / displs / Gatherv sequence one level up -- a prefix sum over
 * per-worker counts is how every level here avoids a sort.
 *
 * Phase C needs a separate scratch array rather than compacting in place:
 * with in-place moves, worker i+1's destination can overlap worker i's
 * source region.
 *
 *
 * ---- Timing ----------------------------------------------------------------
 *
 * The timed region matches the revised Week 4 versions exactly: it starts
 * after buffers are allocated and ends when the sorted list exists in memory
 * on the root. File I/O is excluded.
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
 *        mpicc task2.c -o task2 -O2 -fopenmp -lm
 * Run:   srun ./task2 <n> <threads> [scheme 0|1|2] [label]   (inside a SLURM job)
 *
 * Bind properly once threads are involved, or the measurements are
 * meaningless -- unbound, every rank's threads land on the same cores:
 *        mpirun --map-by socket:PE=<threads> --bind-to core -np <p> ./task2 ...
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
#include <omp.h>

/* Distinct from task1_mpi.c's output_mpi.txt: the two files have to sit side
 * by side so the hybrid result can be diffed against the pure-MPI one. */
#define OUTPUT_FILE "output_hybrid.txt"

/* Candidates per chunk for the cyclic scheme. Matches the pthreads CHUNK
 * and the OpenMP schedule(dynamic, 1000) so all three parallel versions
 * use the same granularity. */
#define CHUNK 1000

/* Upper limit on threads per rank. Bounds the small per-thread arrays and
 * catches a nonsense command line before it becomes a huge allocation. */
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

/* Boundary of thread t's sub-range inside one rank's candidate range
 * [jlo, jhi). Thread t owns [split_boundary(t), split_boundary(t+1)).
 *
 * This is range_boundary() generalised to an interval that does not start at
 * zero. BLOCK splits the candidate count evenly. WEIGHTED equalises estimated
 * cost: the work of trial-dividing everything up to k grows as k^1.5, so the
 * cost of the interval [klo, k) is proportional to k^1.5 - klo^1.5, and
 * setting that to a t/nt fraction of the whole interval's cost gives
 *
 *     k_t = ( klo^1.5 + (t/nt) * (khi^1.5 - klo^1.5) ) ^ (2/3)
 *
 * With klo = 3 this collapses to the k_i = n * (i/p)^(2/3) used at the rank
 * level, so the two levels really are the same rule. */
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

/* Upper bound on how many primes lie in [lo, hi). Used to size each thread's
 * scratch slice without a counting pass over the range.
 *
 * prime_count_bound() bounds pi(x) counting up from zero, which is useless
 * per thread: summing it over the threads over-allocates by roughly the
 * thread count. Rosser & Schoenfeld also give pi(x) > x/ln x for x >= 17, so
 * pi(hi) - pi(lo) is strictly below 1.25506 hi/ln hi - lo/ln lo. Below 17
 * that lower bound does not hold, so fall back to bounding pi(hi) alone.
 * +2 absorbs the truncation and any floating-point wobble. */
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

    //init threads 
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

    /* ---- Rank 0 reads the arguments, then broadcasts them ---------------
     * n = 0 is the "bad input, everyone stop" signal. Without it one rank
     * would exit while the others blocked forever in the broadcast. */
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

    /* The broadcast is serial work in Amdahl's sense: its cost does not fall
     * as ranks are added. Timed so it can be counted in the serial fraction.
     *
     * The thread count rides along with n and the scheme rather than being
     * re-parsed per rank: every rank must agree on it, because it will decide
     * how each rank subdivides its own share of the candidates. Ranks
     * disagreeing here would leave gaps or overlaps in the search. */
    double t_bcast0 = MPI_Wtime();
    MPI_Bcast(params, 3, MPI_INT, 0, MPI_COMM_WORLD);
    double t_bcast = MPI_Wtime() - t_bcast0;

    int n       = params[0];
    int scheme  = params[1];
    int threads = params[2];

    /* Only the root prints, so the label needs no broadcast. */
    const char *label = NULL;
    if (rank == 0 && argc == 5) {
        label = argv[4];
    }

    if (n == 0) {
        MPI_Finalize();
        return 1;
    }

    /* Fix the team size now. Later steps compute each thread's share of the
     * work ahead of time, so the runtime must hand back exactly the number of
     * threads asked for -- never fewer. */
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    long num_candidates = ((long) n - 2) / 2;

    /* ---- Local buffer, sized before the clock starts --------------------
     * Sizing is per thread rather than per rank, since each thread needs its
     * own non-overlapping slice; see the tslice[] loop below. */
    long jlo = 0;
    long jhi = 0;

    /* Chunk-list bookkeeping, cyclic only: this rank owns chunks
     * rank, rank+size, rank+2*size, ... Numbering them m = 0, 1, 2, ... turns
     * the rank's scattered chunks into a contiguous index space that threads
     * can be handed contiguous blocks of. */
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

    /* ---- Level 2: split this rank's share across its threads -------------
     * Computed up front, before the clock starts, so that once the search
     * begins each thread can look up its own bounds with no coordination.
     *
     * tbounds[] is in candidate-index space for the contiguous schemes, and
     * in chunk-list index space (m) for cyclic. */
    long   *tbounds = malloc((size_t) (threads + 1) * sizeof(long));
    long   *tslice  = malloc((size_t) (threads + 1) * sizeof(long));
    int    *tcount  = calloc((size_t) threads + 1, sizeof(int));
    double *ttime   = calloc((size_t) threads, sizeof(double));

    if (tbounds == NULL || tslice == NULL || tcount == NULL || ttime == NULL) {
        fprintf(stderr, "Rank %d: thread metadata allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int t = 0; t <= threads; t++) {
        if (scheme == SCHEME_CYCLIC) {
            tbounds[t] = my_chunks * t / threads;
        } else {
            tbounds[t] = split_boundary(t, threads, jlo, jhi, scheme);
        }
    }

    /* Scratch slice sizes. A thread can never emit more primes than it has
     * candidates, nor more than the prime-counting bound over the values it
     * scans, so the smaller of the two is a safe capacity. tslice[] is the
     * exclusive prefix sum of those capacities: thread t owns
     * scratch[tslice[t] .. tslice[t+1]). */
    tslice[0] = 0;

    for (int t = 0; t < threads; t++) {

        long cand;
        long klo;
        long khi;

        if (scheme == SCHEME_CYCLIC) {
            /* Values touched run from the first chunk this thread owns to the
             * end of its last. The chunks are strided, so that interval is
             * wider than the thread's actual candidate set -- which only makes
             * the bound looser, never wrong. */
            cand = (tbounds[t + 1] - tbounds[t]) * CHUNK;
            klo  = 3 + 2 * (rank + tbounds[t] * size) * CHUNK;
            khi  = (long) n;
        } else {
            cand = tbounds[t + 1] - tbounds[t];
            klo  = 3 + 2 * tbounds[t];
            khi  = 3 + 2 * tbounds[t + 1];
        }

        long bound = prime_count_bound_range(klo, khi);
        long cap   = (cand < bound ? cand : bound);

        tslice[t + 1] = tslice[t] + cap;
    }

    /* +2 covers rank 0's extra entry for the prime 2 and leaves slack for
     * empty ranges. */
    long capacity = tslice[threads] + 2;

    int *local_primes = malloc((size_t) capacity * sizeof(int));
    int *scratch      = malloc((size_t) tslice[threads] * sizeof(int) + 1);

    if (local_primes == NULL || scratch == NULL) {
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

    /* Where the threaded output begins: index 1 on rank 0, which already
     * holds the prime 2, and index 0 everywhere else. */
    int base = local_count;

    #pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num();

        /* omp_get_wtime, not MPI_Wtime: under MPI_THREAD_FUNNELED only the
         * master thread may call into MPI. Both are wall-clock, so the two
         * measurements stay comparable. */
        double t_thread0 = omp_get_wtime();

        int  found = 0;
        int *mine  = scratch + tslice[t];

        /* ---- Phase A: search this thread's own sub-range ----------------
         * Every write lands in this thread's private slice, so there is no
         * lock, no atomic, and no false sharing on the output. */
        if (scheme == SCHEME_CYCLIC) {

            for (long m = tbounds[t]; m < tbounds[t + 1]; m++) {

                long c        = rank + m * size;
                long chunk_lo = c * CHUNK;
                long chunk_hi = chunk_lo + CHUNK;

                if (chunk_hi > num_candidates) {
                    chunk_hi = num_candidates;
                }

                for (long j = chunk_lo; j < chunk_hi; j++) {
                    int k = 3 + 2 * (int) j;
                    if (is_prime(k)) {
                        mine[found++] = k;
                    }
                }
            }

        } else {

            for (long j = tbounds[t]; j < tbounds[t + 1]; j++) {
                int k = 3 + 2 * (int) j;
                if (is_prime(k)) {
                    mine[found++] = k;
                }
            }
        }

        tcount[t + 1] = found;

        /* Recorded before the barrier, so it measures this thread's own
         * search and not the time it then spends waiting for the slowest. */
        ttime[t] = omp_get_wtime() - t_thread0;

        /* ---- Phase B: exclusive prefix sum over the per-thread counts ----
         * Nobody may read tcount[] until every thread has published its own,
         * hence the barrier. omp single then elects one thread for the O(t)
         * scan, and the implicit barrier at the end of single stops the
         * others reading the offsets before they are written. */
        #pragma omp barrier

        #pragma omp single
        {
            for (int i = 0; i < threads; i++) {
                tcount[i + 1] += tcount[i];
            }
        }

        /* ---- Phase C: copy this thread's primes to their final home ------
         * tcount[t] is now the number of primes found by threads below t,
         * which is exactly where this thread's block belongs. Each thread
         * still writes a disjoint destination range, so this is safe to run
         * concurrently. */
        if (found > 0) {
            memcpy(local_primes + base + tcount[t], mine,
                   (size_t) found * sizeof(int));
        }
    }

    local_count = base + tcount[threads];

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

    /* Level-2 imbalance: how unevenly this rank's own threads finished. A
     * rank cannot leave the parallel region until its slowest thread is
     * done, so a large spread here wastes cores even when the ranks are
     * perfectly balanced against one another. Reported separately from the
     * rank-level figure because the two levels can fail independently. */
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

    /* Parallel runtime is the slowest rank, not rank 0. */
    MPI_Reduce(&my_total,  &max_total,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_search, &max_search, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_search, &min_search, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_comm,   &max_comm,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Worst thread imbalance on any rank -- the one that actually held the
     * job up. */
    MPI_Reduce(&my_thread_imb, &max_thread_imb, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

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

        if (n < 100) {
            for (int i = 0; i < total; i++) {
                printf("%d ", all_primes[i]);
            }
            printf("\n");
        } else {
            FILE *fp = fopen(OUTPUT_FILE, "w");
            if (fp == NULL) {
                fprintf(stderr, "Error: could not open %s.\n", OUTPUT_FILE);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            for (int i = 0; i < total; i++) {
                fprintf(fp, "%d\n", all_primes[i]);
            }
            fclose(fp);
        }

        /* Imbalance: 0% means every rank finished searching together, which
         * is the ideal. A large value means ranks sat idle waiting. */
        double imbalance = 0.0;
        if (max_search > 0.0) {
            imbalance = 100.0 * (max_search - min_search) / max_search;
        }

        /* ---- Phase accounting for Amdahl's Law ---------------------------
         * parallel : the search, which divides by procs * threads
         * serial   : broadcast + prefix sum + merge -- work whose cost does
         *            NOT fall as ranks are added
         * overhead : the collective communication, which actually GROWS with
         *            the process count and is the term Amdahl does not model
         *
         * Anything unaccounted for (barrier waits, scheduling jitter) is
         * folded into overhead so the three always sum to the total. */
        double serial_part   = t_bcast + t_prefix + my_merge;
        double parallel_part = max_search;
        double overhead      = max_total - serial_part - parallel_part;
        if (overhead < 0.0) {
            overhead = 0.0;
        }

        printf("scheme=%s n=%d procs=%d threads=%d total_workers=%d "
               "nodes=%d primes=%d\n",
               scheme_name(scheme), n, size, threads, size * threads,
               nodes, total);
        printf("  total time      : %.6f s\n", max_total);
        printf("  parallel (search): %.6f s\n", parallel_part);
        printf("  serial parts     : %.6f s\n", serial_part);
        printf("    broadcast      : %.6f s\n", t_bcast);
        printf("    prefix sum     : %.6f s\n", t_prefix);
        printf("    merge          : %.6f s\n", my_merge);
        printf("  overhead (comm)  : %.6f s\n", overhead);
        printf("    collectives    : %.6f s\n", max_comm);
        printf("  search (fastest) : %.6f s\n", min_search);
        printf("  imbalance (rank) : %.2f %%\n", imbalance);
        printf("  imbalance (thrd) : %.2f %%\n", max_thread_imb);

        /* Unified CSV, same column layout as the serial, pthreads and OpenMP
         * versions so one parser handles every result file.
         * impl,scheme,n,procs,threads,nodes,primes,
         * total,serial,parallel,overhead,imbalance */
        printf("CSV,hybrid,%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f\n",
               label ? label : scheme_name(scheme), n, size, threads, nodes,
               total, max_total, serial_part, parallel_part, overhead,
               imbalance);

        free(all_names);
        free(all_primes);
        free(counts);
        free(displs);
    }

    free(local_primes);
    free(scratch);
    free(tbounds);
    free(tslice);
    free(tcount);
    free(ttime);

    MPI_Finalize();
    return 0;
}