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
 *      root must reassemble p sorted runs afterwards. It knows which rank
 *      owns each chunk, so this takes O(total) with no comparisons between
 *      runs. It is serial work on the root and is measured separately below.
 *      At the thread level the rank's chunks are handed to threads
 *      dynamically, one chunk at a time (see below). Splitting the chunk
 *      list into contiguous per-thread blocks instead would give thread 0
 *      the cheapest chunks and the last thread the dearest -- BLOCK's
 *      imbalance again, one level down.
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
 * So a rank's share is cut into SLOTS: contiguous runs of candidates, each
 * with its own slice of a scratch array and its own prime count, numbered in
 * ascending candidate order.
 *
 *   BLOCK / WEIGHTED  one slot per thread, bounds from split_boundary(),
 *                     handed out schedule(static, 1) so thread t owns slot t
 *   CYCLIC            one slot per chunk the rank owns, handed out
 *                     schedule(dynamic, 1): a thread that finishes a cheap
 *                     chunk immediately takes the next one, so no thread
 *                     idles while its rank still has work -- the same
 *                     balancing schedule(dynamic) gave the Week 4 OpenMP code
 *
 * Then, in three phases:
 *
 *   A  threads search slots, each slot into its own scratch slice
 *   B  the per-slot counts are exclusive-prefix-summed, which yields the
 *      offset in local_primes at which each slot's primes belong
 *   C  each slot's primes are copied to that offset, in parallel
 *
 * Sorted order falls out for free: slots are ascending and each is scanned
 * in ascending order, so slot s's output is entirely below slot s+1's. This
 * is the same prefix-sum compaction the revised Week 4 pthreads and OpenMP
 * versions use, and the same idea as the Gather / displs / Gatherv sequence
 * one level up -- a prefix sum over per-worker counts is how every level
 * here avoids a sort.
 *
 * Phase C needs a separate scratch array rather than compacting in place:
 * with in-place moves, slot s+1's destination can overlap slot s's source.
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
 *        mpicc task2.c -o task2_hybrid -O2 -fopenmp -lm
 * Run:   srun ./task2_hybrid <n> <threads> [scheme 0|1|2] [label]   (inside a SLURM job)
 *
 * On CAAS, allocate the job in the shape it runs, as in the MPI + OpenMP
 * template, and do NOT pin threads:
 *        #SBATCH --ntasks=<p> --cpus-per-task=<threads>
 *        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
 *        srun ./task2_hybrid <n> <threads> ...
 * CAAS's Slurm does not bind ranks to cores, so every rank sees all of the
 * node's CPUs. OMP_PROC_BIND/OMP_PLACES then pins every rank's threads onto
 * the SAME first cores; left unpinned, the Linux scheduler spreads them.
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

/* Scratch capacity for a slot covering candidate indices [lo, hi): the
 * smallest of three upper bounds on the primes it can hold.
 *
 *   - its candidate count
 *   - the Rosser-Schoenfeld range bound, tight for wide slots
 *   - Montgomery & Vaughan (1973): pi(x + y) - pi(x) <= 2y / ln y for y >= 2,
 *     tight for narrow slots. A cyclic chunk spans y = 2 * CHUNK integers,
 *     so at most ~527 primes rather than 1000 -- the Rosser-Schoenfeld
 *     difference is useless there, since it bounds pi(hi) and pi(lo)
 *     separately and their error terms dwarf a 2000-wide gap. */
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

    double y  = 2.0 * (double) cand;
    long   mv = (long) (2.0 * y / log(y)) + 2;
    if (mv < cap) {
        cap = mv;
    }

    return cap;
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

/* Reassemble the CYCLIC scheme's gathered runs into one ascending array.
 * segments[displs[r] .. displs[r]+counts[r]) is run r, rank r's primes in
 * ascending order.
 *
 * No comparisons between runs are needed, because the root already knows
 * which rank owns which part of the number line: chunk c belongs to rank
 * c % p, and covers the candidates k < 3 + 2*(c+1)*CHUNK. So the root walks
 * the chunks in order and, for each one, copies primes from its owner's run
 * until it reaches a prime beyond that chunk. Every prime is copied exactly
 * once, so the cost is O(total + number of chunks) -- independent of p.
 *
 * The earlier version compared the heads of all p runs for every prime,
 * O(total * p), and its cost grew with the process count (0.011 s at p = 1
 * to 0.20 s at p = 32 on CAAS). That broke Amdahl's assumption that the
 * serial part stays constant as processes are added. Only needed for CYCLIC.
 *
 * Rank 0's run also holds the prime 2, which is below chunk 0's limit and is
 * therefore copied first. total_chunks is at least 1 so that this happens
 * even when there are no odd candidates at all (n = 3). */
static void merge_runs(const int *segments, const int *counts, const int *displs,
                       int p, int total, long num_candidates, int *out) {

    int *pos = calloc((size_t) p, sizeof(int));
    if (pos == NULL) {
        fprintf(stderr, "Root: merge allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long total_chunks = (num_candidates + CHUNK - 1) / CHUNK;
    if (total_chunks < 1) {
        total_chunks = 1;
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

    /* Cheap sanity check: every gathered prime must have been placed. */
    if (written != total) {
        fprintf(stderr, "Root: merge placed %d of %d primes.\n", written, total);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
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

    /* ---- Level 1: this rank's share of the candidates --------------------
     * Contiguous schemes own the candidate range [jlo, jhi). Cyclic owns
     * chunks rank, rank+size, rank+2*size, ... -- my_chunks of them. */
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

    /* ---- Level 2: cut the share into slots -------------------------------
     * Computed before the clock starts, so once the search begins a thread
     * only looks bounds up. One slot per chunk for cyclic, one per thread for
     * the contiguous schemes -- see the header for why. */
    long nslots = (scheme == SCHEME_CYCLIC) ? my_chunks : threads;

    long   *slot_lo = malloc((size_t) (nslots + 1) * sizeof(long));
    long   *slot_hi = malloc((size_t) (nslots + 1) * sizeof(long));
    long   *sslice  = malloc((size_t) (nslots + 1) * sizeof(long));
    int    *scount  = calloc((size_t) nslots + 1, sizeof(int));
    double *ttime   = calloc((size_t) threads, sizeof(double));

    if (slot_lo == NULL || slot_hi == NULL || sslice == NULL
        || scount == NULL || ttime == NULL) {
        fprintf(stderr, "Rank %d: slot metadata allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* sslice[] is the exclusive prefix sum of the slot capacities: slot s
     * owns scratch[sslice[s] .. sslice[s+1]). */
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

    /* Static, chunk 1, for one slot per thread: slot t goes to thread t.
     * Dynamic, chunk 1, for one slot per chunk. Set here, so the single
     * schedule(runtime) loop below serves every scheme. */
    if (scheme == SCHEME_CYCLIC) {
        omp_set_schedule(omp_sched_dynamic, 1);
    } else {
        omp_set_schedule(omp_sched_static, 1);
    }

    /* +2 covers rank 0's extra entry for the prime 2 and leaves slack for
     * empty ranges. */
    long capacity = sslice[nslots] + 2;

    int *local_primes = malloc((size_t) capacity * sizeof(int));
    int *scratch      = malloc((size_t) sslice[nslots] * sizeof(int) + 1);

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
        /* omp_get_wtime, not MPI_Wtime: under MPI_THREAD_FUNNELED only the
         * master thread may call into MPI. Both are wall-clock, so the two
         * measurements stay comparable. */
        double t_thread0 = omp_get_wtime();

        /* ---- Phase A: search the slots -----------------------------------
         * Each slot writes only its own scratch slice and its own count, so
         * there is no lock and no atomic. nowait, so the timer below records
         * when THIS thread ran out of work rather than when the slowest did;
         * the end of the parallel region is still a barrier. */
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

    /* ---- Phase B: exclusive prefix sum over the per-slot counts ----------
     * Serial, but only nslots additions -- tens of thousands at most, which
     * is microseconds against a search measured in seconds. */
    for (long s = 0; s < nslots; s++) {
        scount[s + 1] += scount[s];
    }

    /* ---- Phase C: copy each slot's primes to their final home ------------
     * scount[s] is now the number of primes in the slots below s, which is
     * exactly where slot s's block belongs. Destinations are disjoint, so
     * the copies run concurrently. */
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
        merge_runs(all_primes, counts, displs, size, total,
                   num_candidates, merged);
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
         * parallel : the search, which divides by procs * threads
         * serial   : broadcast + prefix sum + merge + file write -- work whose
         *            cost does NOT fall as ranks or threads are added
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

        /* Unified CSV, same column layout as the serial, pthreads and OpenMP
         * versions so one parser handles every result file.
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