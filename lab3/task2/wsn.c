/* FIT3143 Lab 3 - Distributed WSN of EV charging nodes (Open MPI + OpenMP)
 * Charmaine Chee Hing Yi 33520399 | Nazara Glaukofani Lopulisa 35135271
 *
 * compile: mpicc -fopenmp -Wall -o wsn wsn.c
 * run:     mpirun --oversubscribe -np <m*n+1> ./wsn m n [k cycle duration threshold busy_pct
 *                                                      variant period base_variant poll_ms]
 *          e.g. mpirun --oversubscribe -np 10 ./wsn 3 3 6 1.0 30 2 60 0 5.0 0 1.0
 *          low-availability test: busy_pct = 90
 *
 *   variant      node neighbour exchange : 0 Neighbor_allgather | 1 Ineighbor_allgather | 2 Isend/Irecv
 *   period       base station "recent report" window (s)
 *   base_variant base station listener   : 0 Irecv+Test, Isend reply
 *                                          1 Iprobe+Recv, blocking Send reply
 *                                          2 persistent Recv_init per node + Testsome, Isend reply
 *   poll_ms      idle polling interval of the base station and of nodes waiting for the base (ms)
 *
 * world rank 0 = base station; world ranks 1..m*n = charging nodes (cart rank = world rank - 1)
 * output: node_<cart rank>.log, base_<m>x<n>_v.._b.._p.._<timestamp>.log,
 *         results.csv (appended: one row per message type per run, for Task 3) */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

/* ======================= shared definitions ======================= */

#define BASE_RANK   0          /* world rank of the base station        */
#define MAX_PORTS   64
#define MAX_CAND    64         /* max nearest nodes in a redirect reply */
#define HISTORY_LEN 16         /* circular buffer of availability       */
#define SYNC_ROUNDS 8          /* ping-pongs per node for clock sync    */
#define NEVER       (-1e300)   /* "no report yet" timestamp             */

/* neighbour order = MPI_Neighbor_allgather order on a 2D cart comm */
enum { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3, NUM_DIRS = 4 };
static const char *dir_name[NUM_DIRS] = { "up", "down", "left", "right" };

enum {
    TAG_REPORT = 1,   /* node -> base : congestion report              */
    TAG_REPLY  = 2,   /* base -> node : nearest available nodes        */
    TAG_TERM   = 3,   /* base -> node : terminate (payload: send time) */
    TAG_DONE   = 4,   /* node -> base : shut down + node statistics    */
    TAG_NBR    = 5,   /* node <-> node: availability (p2p variant)     */
    TAG_SYNC   = 6    /* node <-> base: clock synchronisation          */
};

/* neighbour-exchange implementations compared in Task 3 */
enum { VAR_NEIGHBOR_ALLGATHER = 0,   /* blocking neighbourhood collective    */
       VAR_INEIGHBOR_ALLGATHER = 1,  /* non-blocking neighbourhood coll.    */
       VAR_P2P_ISEND_IRECV = 2 };    /* non-blocking point-to-point         */

/* base station listener implementations compared in Task 3 */
enum { BV_IRECV_TEST = 0,            /* one Irecv(ANY_SOURCE) + Test, Isend reply       */
       BV_IPROBE_RECV = 1,           /* Iprobe + blocking Recv, blocking Send reply     */
       BV_PERSISTENT_TESTSOME = 2,   /* Recv_init per node + Testsome, Isend reply      */
       NUM_BV = 3 };
static const char *bv_name[NUM_BV] = { "Irecv+Test/Isend", "Iprobe+Recv/Send", "Recv_init+Testsome/Isend" };

enum { REPLY_FOUND = 0,              /* nearest available node(s) listed             */
       REPLY_NONE_NEARBY = 1,        /* all neighbours reported within the period    */
       REPLY_NONE_IN_GRID = 2 };     /* every node is known to be congested          */

typedef struct {
    int m, n, k;             /* grid rows, cols, ports per node              */
    int threshold;           /* congested when free ports < threshold        */
    int busy_pct;            /* chance (%) a port is in use each update      */
    int variant;             /* node neighbour-exchange implementation       */
    int base_variant;        /* base station listener implementation         */
    double cycle;            /* seconds per simulation cycle                 */
    double duration;         /* seconds the base station runs the simulation */
    double period;           /* base: a report is "recent" if younger (s)    */
    double poll_ms;          /* idle polling interval (ms)                   */
} Config;

/* One report = one MPI message. All ints first, then doubles, so the
 * derived datatype is two blocks (see build_report_type). */
typedef struct {
    int cart_rank, world_rank;
    int coords[2];
    int cycle;
    int year, month, day, hour, minute, second;    /* detection time */
    int k, availability;
    int port_state[MAX_PORTS];                      /* 1 = in use     */
    int history[HISTORY_LEN];                       /* recent avail   */
    int num_neighbours;
    int nbr_rank[NUM_DIRS];                         /* cart rank / -1 */
    int nbr_coords[NUM_DIRS][2];
    int nbr_avail[NUM_DIRS];
    int msgs_with_neighbours;                       /* this cycle     */
    int bytes_per_nbr_msg;
    /* doubles */
    double send_wtime;                              /* base clock at send     */
    double nbr_comm_time;                           /* neighbour exchange (s) */
} Report;

typedef struct {
    int status;                      /* REPLY_FOUND / _NONE_NEARBY / _NONE_IN_GRID */
    int count;                       /* nodes listed below (<= MAX_CAND)           */
    int found;                       /* nodes at that distance (may be > count)    */
    int distance;                    /* Manhattan distance (grid hops) of results  */
    int rank[MAX_CAND];              /* cart ranks                                 */
    int coords[MAX_CAND][2];
    double send_wtime;               /* base clock at send                         */
} Reply;

/* running statistics of one measurement (all doubles so it can travel as MPI_DOUBLE) */
typedef struct { double n, sum, min, max; } Acc;

static void acc_init(Acc *a) { a->n = 0; a->sum = 0; a->min = DBL_MAX; a->max = -DBL_MAX; }

static void acc_add(Acc *a, double x) {
    a->n++;
    a->sum += x;
    if (x < a->min) a->min = x;
    if (x > a->max) a->max = x;
}

static void acc_merge(Acc *a, const Acc *b) {
    if (b->n == 0) return;
    a->n += b->n;
    a->sum += b->sum;
    if (b->min < a->min) a->min = b->min;
    if (b->max > a->max) a->max = b->max;
}

/* per-node statistics, sent to the base station in the TAG_DONE message */
typedef struct {
    double cycles, reports, replies, local_redirects, nbr_msgs_sent;
    double term_delay;               /* TERM one-way delay (s)              */
    Acc xchg;                        /* neighbour exchange time per cycle   */
    Acc allreduce;                   /* termination-check Allreduce time    */
    Acc reply;                       /* base -> node reply one-way delay    */
} NodeStats;
#define NODE_STATS_LEN ((int)(sizeof(NodeStats) / sizeof(double)))

/* ======================= MPI derived datatypes ======================= */
/* Report: block of ints followed by block of doubles */
MPI_Datatype build_report_type(void) {
    int n_ints = (int)((offsetof(Report, send_wtime) - offsetof(Report, cart_rank)) / sizeof(int));
    int blocklens[2] = { n_ints, 2 };
    MPI_Aint disps[2] = { offsetof(Report, cart_rank), offsetof(Report, send_wtime) };
    MPI_Datatype types[2] = { MPI_INT, MPI_DOUBLE }, tmp, t;
    MPI_Type_create_struct(2, blocklens, disps, types, &tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(Report), &t);   /* honour trailing padding */
    MPI_Type_free(&tmp);
    MPI_Type_commit(&t);
    return t;
}

MPI_Datatype build_reply_type(void) {
    int n_ints = (int)((offsetof(Reply, send_wtime) - offsetof(Reply, status)) / sizeof(int));
    int blocklens[2] = { n_ints, 1 };
    MPI_Aint disps[2] = { offsetof(Reply, status), offsetof(Reply, send_wtime) };
    MPI_Datatype types[2] = { MPI_INT, MPI_DOUBLE }, tmp, t;
    MPI_Type_create_struct(2, blocklens, disps, types, &tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(Reply), &t);
    MPI_Type_free(&tmp);
    MPI_Type_commit(&t);
    return t;
}

static int arg_i(int argc, char **argv, int i, int def) { return argc > i ? atoi(argv[i]) : def; }
static double arg_d(int argc, char **argv, int i, double def) { return argc > i ? atof(argv[i]) : def; }

static void sleep_seconds(double s) {
    if (s <= 0) return;
    struct timespec ts;
    ts.tv_sec = (time_t)s;
    ts.tv_nsec = (long)((s - (double)ts.tv_sec) * 1e9);
    nanosleep(&ts, NULL);
}

/* ======================= clock synchronisation =======================
 * Nodes may run on different CAAS hosts whose MPI_Wtime clocks are not aligned,
 * so "receive time - send time" across processes is meaningless without this.
 * Cristian's algorithm: base_clock ~= MPI_Wtime() + offset. The round with the
 * smallest RTT is kept, so the offset error is at most RTT_min / 2. */
static double clock_sync_node(void) {
    double best_rtt = DBL_MAX, offset = 0;
    for (int r = 0; r < SYNC_ROUNDS; r++) {
        double t0 = MPI_Wtime(), tb;
        MPI_Send(&t0, 1, MPI_DOUBLE, BASE_RANK, TAG_SYNC, MPI_COMM_WORLD);
        MPI_Recv(&tb, 1, MPI_DOUBLE, BASE_RANK, TAG_SYNC, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double t1 = MPI_Wtime();
        if (t1 - t0 < best_rtt) {
            best_rtt = t1 - t0;
            offset = tb - (t0 + t1) / 2;
        }
    }
    return offset;
}

static void clock_sync_base(int num_nodes) {
    for (int i = 1; i <= num_nodes; i++)
        for (int r = 0; r < SYNC_ROUNDS; r++) {
            double t;
            MPI_Recv(&t, 1, MPI_DOUBLE, i, TAG_SYNC, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            t = MPI_Wtime();
            MPI_Send(&t, 1, MPI_DOUBLE, i, TAG_SYNC, MPI_COMM_WORLD);
        }
}

/* ======================= TASK 1: EV charging node ======================= */
/* ---------- state shared by all threads of this process ---------- */
static int        port_state[MAX_PORTS];     /* 1 = in use, 0 = free      */
static int        history[HISTORY_LEN];      /* availability per cycle    */
static int        history_count = 0;
static omp_lock_t state_lock;                /* guards port_state/history */
static int        g_terminate = 0;           /* set by comm thread        */
static double     g_clock_offset = 0;        /* base clock - local clock  */

static double base_clock(void) { return MPI_Wtime() + g_clock_offset; }

static int should_stop(void) {
    int t;
    #pragma omp atomic read
    t = g_terminate;
    return t;
}

/* ---------- charging port thread ---------- */
static void port_thread(const Config *cfg, int port, int cart_rank) {
    unsigned int seed = (unsigned int)(time(NULL) ^ (cart_rank * 7919) ^ (port * 104729));
    /* stagger ports so they do not all change in lock-step */
    sleep_seconds(cfg->cycle * port / (double)cfg->k);
    while (!should_stop()) {
        /* EV arrives / departs: port is in use with probability busy_pct */
        int in_use = (rand_r(&seed) % 100) < cfg->busy_pct;
        omp_set_lock(&state_lock);
        port_state[port] = in_use;
        omp_unset_lock(&state_lock);

        /* sleep one cycle in small slices so termination is noticed quickly */
        for (double slept = 0; slept < cfg->cycle && !should_stop(); slept += 0.05)
            sleep_seconds(0.05);
    }
}

/* ---------- helpers for the comm thread ---------- */
static int snapshot_availability(int k, int *states_out) {
    int free_ports = 0;
    omp_set_lock(&state_lock);                /* consistent snapshot */
    for (int i = 0; i < k; i++) {
        states_out[i] = port_state[i];
        free_ports += !port_state[i];
    }
    history[history_count % HISTORY_LEN] = free_ports;
    history_count++;
    omp_unset_lock(&state_lock);
    return free_ports;
}

/* exchange my availability with the 4 neighbours; returns real neighbour count.
 * nbr_avail[d] = -1 for a missing (MPI_PROC_NULL) neighbour. */
static int exchange_with_neighbours(const Config *cfg, MPI_Comm cart, const int nbr[NUM_DIRS],
                                    int my_avail, int nbr_avail[NUM_DIRS]) {
    for (int d = 0; d < NUM_DIRS; d++) nbr_avail[d] = -1;   /* PROC_NULL slots stay -1 */

    if (cfg->variant == VAR_NEIGHBOR_ALLGATHER) {
        MPI_Neighbor_allgather(&my_avail, 1, MPI_INT, nbr_avail, 1, MPI_INT, cart);
    } else if (cfg->variant == VAR_INEIGHBOR_ALLGATHER) {
        MPI_Request req;
        MPI_Ineighbor_allgather(&my_avail, 1, MPI_INT, nbr_avail, 1, MPI_INT, cart, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else {   /* VAR_P2P_ISEND_IRECV: sends/recvs to MPI_PROC_NULL are no-ops */
        MPI_Request reqs[2 * NUM_DIRS];
        for (int d = 0; d < NUM_DIRS; d++) {
            MPI_Irecv(&nbr_avail[d], 1, MPI_INT, nbr[d], TAG_NBR, cart, &reqs[d]);
            MPI_Isend(&my_avail, 1, MPI_INT, nbr[d], TAG_NBR, cart, &reqs[NUM_DIRS + d]);
        }
        MPI_Waitall(2 * NUM_DIRS, reqs, MPI_STATUSES_IGNORE);
    }

    int count = 0;
    for (int d = 0; d < NUM_DIRS; d++) count += (nbr[d] != MPI_PROC_NULL);
    return count;
}

/* messages from the base station that the comm thread keeps a receive posted for */
typedef struct {
    MPI_Request term_req, reply_req;
    double term_send;          /* TERM payload: base clock at send */
    int term_flag;
    Reply reply;
} BaseLink;

static void log_reply(FILE *log, const Reply *r, double delay) {
    if (r->status == REPLY_FOUND) {
        fprintf(log, "  [base reply] forward EVs to %d node(s) at distance %d:", r->count, r->distance);
        for (int i = 0; i < r->count; i++)
            fprintf(log, " node%d(%d,%d)", r->rank[i], r->coords[i][0], r->coords[i][1]);
    } else if (r->status == REPLY_NONE_NEARBY) {
        fprintf(log, "  [base reply] no nodes available nearby");
    } else {
        fprintf(log, "  [base reply] no available node in the whole grid");
    }
    fprintf(log, " (delay %.3f ms)\n", delay * 1e3);
}

/* non-blocking check for a reply and for the termination message */
static void poll_base(BaseLink *bl, NodeStats *st, FILE *log) {
    if (!bl->term_flag) {
        MPI_Test(&bl->term_req, &bl->term_flag, MPI_STATUS_IGNORE);
        if (bl->term_flag) st->term_delay = base_clock() - bl->term_send;
    }
    if (bl->reply_req != MPI_REQUEST_NULL) {
        int done = 0;
        MPI_Test(&bl->reply_req, &done, MPI_STATUS_IGNORE);
        if (done) {
            double delay = base_clock() - bl->reply.send_wtime;
            st->replies++;
            acc_add(&st->reply, delay);
            log_reply(log, &bl->reply, delay);
        }
    }
}

/* ---------- communication thread (master) ---------- */
static void comm_thread(const Config *cfg, MPI_Comm cart, MPI_Datatype report_t,
                        MPI_Datatype reply_t, FILE *log) {
    int cart_rank, coords[2], nbr[NUM_DIRS], nbr_coords[NUM_DIRS][2];
    MPI_Comm_rank(cart, &cart_rank);
    MPI_Cart_coords(cart, cart_rank, 2, coords);
    MPI_Cart_shift(cart, 0, 1, &nbr[UP],   &nbr[DOWN]);
    MPI_Cart_shift(cart, 1, 1, &nbr[LEFT], &nbr[RIGHT]);
    for (int d = 0; d < NUM_DIRS; d++) {
        nbr_coords[d][0] = nbr_coords[d][1] = -1;
        if (nbr[d] != MPI_PROC_NULL) MPI_Cart_coords(cart, nbr[d], 2, nbr_coords[d]);
    }

    fprintf(log, "node cart=%d world=%d coords=(%d,%d) k=%d threshold=%d variant=%d clock_offset=%.6f s\n",
            cart_rank, cart_rank + 1, coords[0], coords[1], cfg->k, cfg->threshold, cfg->variant,
            g_clock_offset);
    for (int d = 0; d < NUM_DIRS; d++) {
        if (nbr[d] == MPI_PROC_NULL) fprintf(log, "  neighbour %-5s : none\n", dir_name[d]);
        else fprintf(log, "  neighbour %-5s : node %d (%d,%d)\n", dir_name[d], nbr[d],
                     nbr_coords[d][0], nbr_coords[d][1]);
    }
    fprintf(log, "YYYY MM DD HH MM SS | avail | ports(1=used) | nbr avail u/d/l/r | xchg(ms) | action\n");

    /* termination message can arrive at any time: keep a receive posted */
    BaseLink bl;
    memset(&bl, 0, sizeof bl);
    bl.reply_req = MPI_REQUEST_NULL;
    MPI_Request report_req = MPI_REQUEST_NULL;
    MPI_Irecv(&bl.term_send, 1, MPI_DOUBLE, BASE_RANK, TAG_TERM, MPI_COMM_WORLD, &bl.term_req);

    Report    report;
    int       states[MAX_PORTS], nbr_avail[NUM_DIRS];
    NodeStats st;
    memset(&st, 0, sizeof st);
    acc_init(&st.xchg);
    acc_init(&st.allreduce);
    acc_init(&st.reply);
    double poll = cfg->poll_ms / 1e3;

    for (int cycle = 0; ; cycle++) {
        double cycle_start = MPI_Wtime();

        /* 1. termination: all nodes must leave the loop in the SAME cycle,
         *    otherwise a node would block in a neighbour exchange forever */
        poll_base(&bl, &st, log);
        int stop = 0;
        double t0 = MPI_Wtime();
        MPI_Allreduce(&bl.term_flag, &stop, 1, MPI_INT, MPI_MAX, cart);
        acc_add(&st.allreduce, MPI_Wtime() - t0);
        if (stop) break;
        st.cycles++;

        /* 2. read shared array */
        int avail = snapshot_availability(cfg->k, states);
        time_t now = time(NULL);
        struct tm tmv;
        localtime_r(&now, &tmv);

        /* 3. neighbour exchange (timed) */
        t0 = MPI_Wtime();
        int num_nbrs = exchange_with_neighbours(cfg, cart, nbr, avail, nbr_avail);
        double xchg = MPI_Wtime() - t0;
        acc_add(&st.xchg, xchg);
        st.nbr_msgs_sent += num_nbrs;

        /* 4. decision */
        char action[96] = "ok";
        if (avail < cfg->threshold) {
            int best = -1;
            for (int d = 0; d < NUM_DIRS; d++)
                if (nbr_avail[d] >= cfg->threshold && (best < 0 || nbr_avail[d] > nbr_avail[best]))
                    best = d;

            if (best >= 0) {
                /* a neighbour has room: redirect locally, no base station traffic */
                st.local_redirects++;
                snprintf(action, sizeof action, "CONGESTED -> redirect EVs to node %d (%d,%d) avail=%d",
                         nbr[best], nbr_coords[best][0], nbr_coords[best][1], nbr_avail[best]);
            } else if (bl.reply_req != MPI_REQUEST_NULL) {
                snprintf(action, sizeof action, "CONGESTED, neighbours full, awaiting base reply");
            } else {
                /* node and all neighbours congested: alert the base station */
                MPI_Wait(&report_req, MPI_STATUS_IGNORE);   /* report buffer reusable */
                memset(&report, 0, sizeof report);
                report.cart_rank = cart_rank;
                report.world_rank = cart_rank + 1;
                report.coords[0] = coords[0]; report.coords[1] = coords[1];
                report.cycle = cycle;
                report.year = tmv.tm_year + 1900; report.month = tmv.tm_mon + 1; report.day = tmv.tm_mday;
                report.hour = tmv.tm_hour; report.minute = tmv.tm_min; report.second = tmv.tm_sec;
                report.k = cfg->k;
                report.availability = avail;
                memcpy(report.port_state, states, sizeof(int) * cfg->k);
                omp_set_lock(&state_lock);
                for (int i = 0; i < HISTORY_LEN; i++) {     /* oldest -> newest, -1 = unused */
                    int idx = history_count - HISTORY_LEN + i;
                    report.history[i] = idx >= 0 ? history[idx % HISTORY_LEN] : -1;
                }
                omp_unset_lock(&state_lock);
                report.num_neighbours = num_nbrs;
                for (int d = 0; d < NUM_DIRS; d++) {
                    report.nbr_rank[d] = nbr[d] == MPI_PROC_NULL ? -1 : nbr[d];
                    report.nbr_coords[d][0] = nbr_coords[d][0];
                    report.nbr_coords[d][1] = nbr_coords[d][1];
                    report.nbr_avail[d] = nbr_avail[d];
                }
                report.msgs_with_neighbours = 2 * num_nbrs;     /* one sent + one received each */
                report.bytes_per_nbr_msg = (int)sizeof(int);
                report.nbr_comm_time = xchg;

                /* post the reply receive BEFORE sending, then send one message */
                MPI_Irecv(&bl.reply, 1, reply_t, BASE_RANK, TAG_REPLY, MPI_COMM_WORLD, &bl.reply_req);
                report.send_wtime = base_clock();
                MPI_Isend(&report, 1, report_t, BASE_RANK, TAG_REPORT, MPI_COMM_WORLD, &report_req);
                st.reports++;
                snprintf(action, sizeof action, "CONGESTED, all neighbours full -> ALERT base station");
            }
        }

        fprintf(log, "%04d %02d %02d %02d %02d %02d | %5d | ",
                tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec, avail);
        for (int i = 0; i < cfg->k; i++) fputc(states[i] ? '1' : '0', log);
        fprintf(log, " | %2d %2d %2d %2d | %8.3f | %s\n",
                nbr_avail[UP], nbr_avail[DOWN], nbr_avail[LEFT], nbr_avail[RIGHT], xchg * 1e3, action);

        /* 5. rest of the cycle: poll the base station every poll_ms so a reply is
         *    forwarded to EVs (and timestamped) promptly instead of next cycle */
        double left;
        while ((left = cfg->cycle - (MPI_Wtime() - cycle_start)) > 0) {
            poll_base(&bl, &st, log);
            sleep_seconds(left < poll ? left : poll);
        }
    }

    /* ---------- graceful shutdown of communication ---------- */
    #pragma omp atomic write
    g_terminate = 1;                                   /* port threads exit their loops */

    if (!bl.term_flag) {                               /* base sent it to everyone */
        MPI_Wait(&bl.term_req, MPI_STATUS_IGNORE);
        bl.term_flag = 1;
        st.term_delay = base_clock() - bl.term_send;
    }
    MPI_Wait(&report_req, MPI_STATUS_IGNORE);
    if (bl.reply_req != MPI_REQUEST_NULL) {            /* reply that will never come */
        MPI_Status s;
        int cancelled;
        MPI_Cancel(&bl.reply_req);
        MPI_Wait(&bl.reply_req, &s);
        MPI_Test_cancelled(&s, &cancelled);
        if (!cancelled) {                              /* it did arrive before the cancel */
            double delay = base_clock() - bl.reply.send_wtime;
            st.replies++;
            acc_add(&st.reply, delay);
            log_reply(log, &bl.reply, delay);
        }
    }
    /* tell the base station we are done (with our statistics);
     * it drains reports until all DONEs arrive */
    MPI_Send(&st, NODE_STATS_LEN, MPI_DOUBLE, BASE_RANK, TAG_DONE, MPI_COMM_WORLD);

    fprintf(log, "\nTERMINATED by base station (termination delay %.3f ms)\n", st.term_delay * 1e3);
    fprintf(log, "summary: cycles=%.0f reports_to_base=%.0f replies=%.0f local_redirects=%.0f "
            "avg_xchg=%.3f ms avg_reply_delay=%.3f ms\n",
            st.cycles, st.reports, st.replies, st.local_redirects,
            st.xchg.n ? st.xchg.sum / st.xchg.n * 1e3 : 0.0,
            st.reply.n ? st.reply.sum / st.reply.n * 1e3 : 0.0);
}

void node_run(const Config *cfg, MPI_Comm cart, MPI_Datatype report_t, MPI_Datatype reply_t,
              double clock_offset) {
    int cart_rank;
    MPI_Comm_rank(cart, &cart_rank);

    char fname[64];
    snprintf(fname, sizeof fname, "node_%d.log", cart_rank);
    FILE *log = fopen(fname, "w");
    if (!log) log = stderr;

    omp_init_lock(&state_lock);
    memset(port_state, 0, sizeof port_state);
    g_terminate = 0;
    g_clock_offset = clock_offset;

    #pragma omp parallel num_threads(cfg->k + 1)
    {
        int tid = omp_get_thread_num();
        if (tid == 0) comm_thread(cfg, cart, report_t, reply_t, log);   /* master = FUNNELED */
        else          port_thread(cfg, tid - 1, cart_rank);
    }   /* implicit barrier: every port thread has returned */

    omp_destroy_lock(&state_lock);
    if (log != stderr) fclose(log);
}

/* ======================= TASK 2: base station =======================
 * One MPI process, one thread (no OpenMP team): the only thread that sends or
 * receives, so no thread creation/destruction or MPI thread-safety overhead.
 *
 * Knowledge table (base clock):
 *   last_report[j] - when node j last reported congestion
 *   last_full[j]   - when node j was last KNOWN to be full: it reported, or it
 *                    appeared as a full neighbour inside someone's report
 *                    (a report from node 1 also covers nodes 0, 2, 4)
 * On a report from node X:
 *   - every neighbour of X reported within `period`  -> "no nodes available nearby"
 *   - otherwise return ALL nodes at the smallest grid distance from X that are
 *     not known full within `period` (ring 2 = neighbours of neighbours first,
 *     expanding outwards if that ring is congested too). */
typedef struct {
    const Config *cfg;
    int           N;
    FILE         *log;
    MPI_Datatype  reply_t;
    double        start;
    double       *last_report;
    double       *last_full;
    int          *alerts_by_node;
    Reply        *reply_buf;           /* one per node: a node has <= 1 report outstanding */
    MPI_Request  *reply_req;
    int           alerts, late_reports, replies_found, replies_none;
    Acc           report_delay;        /* node -> base one-way delay                 */
    Acc           proc_time;           /* receive -> reply send at the base          */
} Base;

static int find_nearest(const Base *b, const Report *rep, double now,
                        const double nbr_age[NUM_DIRS], Reply *r) {
    const Config *cfg = b->cfg;
    int n = cfg->n, r0 = rep->cart_rank / n, c0 = rep->cart_rank % n;   /* row-major, reorder = 0 */
    int nbrs = 0, recent = 0;
    for (int d = 0; d < NUM_DIRS; d++) {
        if (rep->nbr_rank[d] < 0) continue;
        nbrs++;
        if (nbr_age[d] <= cfg->period) recent++;
    }

    memset(r, 0, sizeof *r);
    if (recent == nbrs) {
        r->status = REPLY_NONE_NEARBY;
        return recent;
    }

    /* nearest = smallest Manhattan distance (hops in the mesh) among nodes not known full */
    int best = INT_MAX;
    for (int j = 0; j < b->N; j++) {
        int dist = abs(j / n - r0) + abs(j % n - c0);
        if (dist > 0 && dist < best && now - b->last_full[j] > cfg->period) best = dist;
    }
    if (best == INT_MAX) {
        r->status = REPLY_NONE_IN_GRID;
        return recent;
    }

    r->status = REPLY_FOUND;
    r->distance = best;
    for (int j = 0; j < b->N; j++) {   /* report ALL nodes tied at the nearest distance */
        int dist = abs(j / n - r0) + abs(j % n - c0);
        if (dist != best || now - b->last_full[j] <= cfg->period) continue;
        if (r->count < MAX_CAND) {
            r->rank[r->count] = j;
            r->coords[r->count][0] = j / n;
            r->coords[r->count][1] = j % n;
            r->count++;
        }
        r->found++;
    }
    return recent;
}

static void log_report(const Base *b, const Report *rep, double recv_time, double delay,
                       const double nbr_age[NUM_DIRS], int recent, const Reply *r) {
    FILE *f = b->log;
    fprintf(f, "[t=%9.3fs] ALERT #%d from node %d (world %d) at (%d,%d), cycle %d, "
            "detected %04d-%02d-%02d %02d:%02d:%02d\n",
            recv_time - b->start, b->alerts, rep->cart_rank, rep->world_rank, rep->coords[0],
            rep->coords[1], rep->cycle, rep->year, rep->month, rep->day, rep->hour, rep->minute,
            rep->second);

    fprintf(f, "    availability %d/%d free, ports [", rep->availability, rep->k);
    for (int i = 0; i < rep->k && i < MAX_PORTS; i++) fputc(rep->port_state[i] ? '1' : '0', f);
    fprintf(f, "] (1=in use), recent availability:");
    for (int i = 0; i < HISTORY_LEN; i++)
        if (rep->history[i] >= 0) fprintf(f, " %d", rep->history[i]);
    fputc('\n', f);

    fprintf(f, "    neighbours (%d):", rep->num_neighbours);
    int nbrs = 0;
    for (int d = 0; d < NUM_DIRS; d++) {
        if (rep->nbr_rank[d] < 0) continue;
        nbrs++;
        fprintf(f, " %s=node%d(%d,%d) avail=%d last_report=", dir_name[d], rep->nbr_rank[d],
                rep->nbr_coords[d][0], rep->nbr_coords[d][1], rep->nbr_avail[d]);
        if (nbr_age[d] > 1e200) fprintf(f, "never");
        else fprintf(f, "%.2fs ago", nbr_age[d]);
    }
    fputc('\n', f);

    fprintf(f, "    neighbour exchange: %d msgs x %d B in %.3f ms | report delay %.3f ms\n",
            rep->msgs_with_neighbours, rep->bytes_per_nbr_msg, rep->nbr_comm_time * 1e3, delay * 1e3);

    fprintf(f, "    decision: %d/%d neighbours reported within %.2fs -> ", recent, nbrs, b->cfg->period);
    if (r->status == REPLY_NONE_NEARBY) {
        fprintf(f, "no nodes available nearby");
    } else if (r->status == REPLY_NONE_IN_GRID) {
        fprintf(f, "no available node in the whole grid");
    } else {
        fprintf(f, "%d nearest available node(s) at distance %d:", r->found, r->distance);
        for (int i = 0; i < r->count; i++)
            fprintf(f, " node%d(%d,%d)", r->rank[i], r->coords[i][0], r->coords[i][1]);
        if (r->count < r->found) fprintf(f, " (first %d listed)", r->count);
    }
    fprintf(f, " | processing %.3f ms\n", (r->send_wtime - recv_time) * 1e3);
}

static void handle_report(Base *b, const Report *rep, double recv_time) {
    const Config *cfg = b->cfg;
    int me = rep->cart_rank;
    double delay = recv_time - rep->send_wtime;      /* both on the base clock */
    b->alerts++;
    b->alerts_by_node[me]++;
    acc_add(&b->report_delay, delay);

    /* neighbours' report ages, read before this report touches the tables */
    double nbr_age[NUM_DIRS];
    for (int d = 0; d < NUM_DIRS; d++)
        nbr_age[d] = rep->nbr_rank[d] >= 0 ? recv_time - b->last_report[rep->nbr_rank[d]] : -1;

    /* this report proves the node and its reported neighbours are full now */
    b->last_report[me] = b->last_full[me] = recv_time;
    for (int d = 0; d < NUM_DIRS; d++)
        if (rep->nbr_rank[d] >= 0 && rep->nbr_avail[d] >= 0 && rep->nbr_avail[d] < cfg->threshold)
            b->last_full[rep->nbr_rank[d]] = recv_time;

    /* this node's previous reply must have left its buffer before reuse */
    if (b->reply_req[me] != MPI_REQUEST_NULL) MPI_Wait(&b->reply_req[me], MPI_STATUS_IGNORE);
    Reply *r = &b->reply_buf[me];
    int recent = find_nearest(b, rep, recv_time, nbr_age, r);

    r->send_wtime = MPI_Wtime();
    acc_add(&b->proc_time, r->send_wtime - recv_time);
    if (cfg->base_variant == BV_IPROBE_RECV)
        MPI_Send(r, 1, b->reply_t, me + 1, TAG_REPLY, MPI_COMM_WORLD);
    else
        MPI_Isend(r, 1, b->reply_t, me + 1, TAG_REPLY, MPI_COMM_WORLD, &b->reply_req[me]);
    if (r->status == REPLY_FOUND) b->replies_found++;
    else b->replies_none++;

    /* file I/O after the reply is on its way, so it is not on the critical path */
    log_report(b, rep, recv_time, delay, nbr_age, recent, r);
}

static void log_late(Base *b, const Report *rep) {
    b->late_reports++;
    fprintf(b->log, "[t=%9.3fs] late report from node %d (%d,%d) after listening stopped - not answered\n",
            MPI_Wtime() - b->start, rep->cart_rank, rep->coords[0], rep->coords[1]);
}

static void log_acc(FILE *f, const char *name, const Acc *a) {
    if (a->n == 0) fprintf(f, "  %-26s %8d %10s %10s %10s\n", name, 0, "-", "-", "-");
    else fprintf(f, "  %-26s %8.0f %10.3f %10.3f %10.3f\n", name, a->n,
                 a->sum / a->n * 1e3, a->min * 1e3, a->max * 1e3);
}

static void csv_row(FILE *f, const Config *cfg, const char *type, const Acc *a, int alerts,
                    double sim_time) {
    fprintf(f, "%s,%d,%d,%d,%d,%.3f,%d,%d,%d,%d,%.3f,%.3f,%s,%.0f,",
            (cfg->m == 1 || cfg->n == 1) ? "line" : "grid", cfg->m, cfg->n, cfg->m * cfg->n, cfg->k,
            cfg->cycle, cfg->threshold, cfg->busy_pct, cfg->variant, cfg->base_variant,
            cfg->period, cfg->poll_ms, type, a->n);
    if (a->n > 0) fprintf(f, "%.6f,%.6f,%.6f", a->sum / a->n * 1e3, a->min * 1e3, a->max * 1e3);
    else fprintf(f, ",,");
    fprintf(f, ",%d,%.3f\n", alerts, sim_time);
}

void base_run(const Config *cfg, MPI_Datatype report_t, MPI_Datatype reply_t) {
    int N = cfg->m * cfg->n;
    Base b;
    memset(&b, 0, sizeof b);
    b.cfg = cfg;
    b.N = N;
    b.reply_t = reply_t;
    b.last_report    = malloc(N * sizeof(double));
    b.last_full      = malloc(N * sizeof(double));
    b.alerts_by_node = calloc(N, sizeof(int));
    b.reply_buf      = calloc(N, sizeof(Reply));
    b.reply_req      = malloc(N * sizeof(MPI_Request));
    if (!b.last_report || !b.last_full || !b.alerts_by_node || !b.reply_buf || !b.reply_req) {
        fprintf(stderr, "base station: out of memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < N; i++) {
        b.last_report[i] = b.last_full[i] = NEVER;
        b.reply_req[i] = MPI_REQUEST_NULL;
    }
    acc_init(&b.report_delay);
    acc_init(&b.proc_time);

    /* one log per run, so a batch of CAAS runs never overwrites earlier logs */
    char fname[160], stamp[32];
    time_t now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", &tmv);
    snprintf(fname, sizeof fname, "base_%dx%d_v%d_b%d_p%g_%s.log",
             cfg->m, cfg->n, cfg->variant, cfg->base_variant, cfg->period, stamp);
    b.log = fopen(fname, "w");
    if (!b.log) b.log = stderr;

    int report_bytes, reply_bytes;
    MPI_Type_size(report_t, &report_bytes);
    MPI_Type_size(reply_t, &reply_bytes);
    fprintf(b.log, "BASE STATION started %04d-%02d-%02d %02d:%02d:%02d\n",
            tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    fprintf(b.log, "grid=%dx%d (%d nodes, %s) k=%d threshold=%d busy=%d%% cycle=%.2fs duration=%.1fs\n",
            cfg->m, cfg->n, N, (cfg->m == 1 || cfg->n == 1) ? "line" : "grid", cfg->k, cfg->threshold,
            cfg->busy_pct, cfg->cycle, cfg->duration);
    fprintf(b.log, "node exchange variant=%d, base listener=%d (%s), period=%.2fs, poll=%.2fms\n",
            cfg->variant, cfg->base_variant, bv_name[cfg->base_variant], cfg->period, cfg->poll_ms);
    fprintf(b.log, "message sizes: report %d B, reply %d B, termination %d B\n\n",
            report_bytes, reply_bytes, (int)sizeof(double));

    Report      rep;                 /* variants 0, 1 */
    Report     *prep = NULL;         /* variant 2: one receive buffer per node */
    MPI_Request req = MPI_REQUEST_NULL, *preq = NULL;
    int        *idx = NULL;
    MPI_Status  st;
    double      poll = cfg->poll_ms / 1e3;

    b.start = MPI_Wtime();
    if (cfg->base_variant == BV_IRECV_TEST) {
        MPI_Irecv(&rep, 1, report_t, MPI_ANY_SOURCE, TAG_REPORT, MPI_COMM_WORLD, &req);
    } else if (cfg->base_variant == BV_PERSISTENT_TESTSOME) {
        prep = malloc(N * sizeof(Report));
        preq = malloc(N * sizeof(MPI_Request));
        idx  = malloc(N * sizeof(int));
        if (!prep || !preq || !idx) {
            fprintf(stderr, "base station: out of memory\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < N; i++)
            MPI_Recv_init(&prep[i], 1, report_t, i + 1, TAG_REPORT, MPI_COMM_WORLD, &preq[i]);
        MPI_Startall(N, preq);
    }

    /* ---------- periodically listen for reports until the simulation ends ---------- */
    while (MPI_Wtime() - b.start < cfg->duration) {
        int got = 0;
        if (cfg->base_variant == BV_IRECV_TEST) {
            MPI_Test(&req, &got, MPI_STATUS_IGNORE);
            if (got) {
                handle_report(&b, &rep, MPI_Wtime());
                MPI_Irecv(&rep, 1, report_t, MPI_ANY_SOURCE, TAG_REPORT, MPI_COMM_WORLD, &req);
            }
        } else if (cfg->base_variant == BV_IPROBE_RECV) {
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_REPORT, MPI_COMM_WORLD, &got, &st);
            if (got) {
                MPI_Recv(&rep, 1, report_t, st.MPI_SOURCE, TAG_REPORT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                handle_report(&b, &rep, MPI_Wtime());
            }
        } else {
            /* all reports that arrived since the last poll are handled in one batch */
            int outcount;
            MPI_Testsome(N, preq, &outcount, idx, MPI_STATUSES_IGNORE);
            if (outcount == MPI_UNDEFINED) outcount = 0;
            double t = MPI_Wtime();
            for (int i = 0; i < outcount; i++) {
                handle_report(&b, &prep[idx[i]], t);
                MPI_Start(&preq[idx[i]]);
            }
            got = outcount > 0;
        }
        if (!got) sleep_seconds(poll);   /* idle: yield the CPU instead of busy-waiting */
    }
    double listen_end = MPI_Wtime();

    /* ---------- stop listening: cancel posted receives ---------- */
    if (cfg->base_variant == BV_IRECV_TEST) {
        int cancelled;
        MPI_Cancel(&req);
        MPI_Wait(&req, &st);
        MPI_Test_cancelled(&st, &cancelled);
        if (!cancelled) log_late(&b, &rep);          /* it matched a report just in time */
    } else if (cfg->base_variant == BV_PERSISTENT_TESTSOME) {
        for (int i = 0; i < N; i++) {
            int cancelled;
            MPI_Cancel(&preq[i]);
            MPI_Wait(&preq[i], &st);
            MPI_Test_cancelled(&st, &cancelled);
            if (!cancelled) log_late(&b, &prep[i]);
            MPI_Request_free(&preq[i]);
        }
    }

    /* ---------- termination message to every node ----------
     * point-to-point: the base station is not in the cart communicator */
    double term_send = MPI_Wtime();
    MPI_Request *treq = malloc(N * sizeof(MPI_Request));
    if (!treq) MPI_Abort(MPI_COMM_WORLD, 1);
    for (int i = 0; i < N; i++)
        MPI_Isend(&term_send, 1, MPI_DOUBLE, i + 1, TAG_TERM, MPI_COMM_WORLD, &treq[i]);
    MPI_Waitall(N, treq, MPI_STATUSES_IGNORE);

    /* ---------- drain in-flight reports until every node confirms shutdown ---------- */
    NodeStats ns, tot;
    memset(&tot, 0, sizeof tot);
    acc_init(&tot.xchg);
    acc_init(&tot.allreduce);
    acc_init(&tot.reply);
    Acc term_delay;
    acc_init(&term_delay);
    int done = 0;
    while (done < N) {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
        if (st.MPI_TAG == TAG_DONE) {
            MPI_Recv(&ns, NODE_STATS_LEN, MPI_DOUBLE, st.MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            tot.cycles += ns.cycles;
            tot.reports += ns.reports;
            tot.replies += ns.replies;
            tot.local_redirects += ns.local_redirects;
            tot.nbr_msgs_sent += ns.nbr_msgs_sent;
            acc_add(&term_delay, ns.term_delay);
            acc_merge(&tot.xchg, &ns.xchg);
            acc_merge(&tot.allreduce, &ns.allreduce);
            acc_merge(&tot.reply, &ns.reply);
            done++;
        } else {
            MPI_Recv(&rep, 1, report_t, st.MPI_SOURCE, TAG_REPORT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            log_late(&b, &rep);
        }
    }
    MPI_Waitall(N, b.reply_req, MPI_STATUSES_IGNORE);   /* every Isend reply has completed */
    double sim_time = MPI_Wtime() - b.start;

    /* ---------- key performance metrics ---------- */
    long long sync_msgs = 2LL * SYNC_ROUNDS * N;
    long long reports = b.alerts + b.late_reports;
    long long p2p_total = sync_msgs + (long long)tot.nbr_msgs_sent + reports + b.alerts + 2LL * N;
    FILE *f = b.log;
    fprintf(f, "\n================ SUMMARY ================\n");
    fprintf(f, "simulation time      : %.3f s (listening %.3f s, shutdown %.3f s)\n",
            sim_time, listen_end - b.start, sim_time - (listen_end - b.start));
    fprintf(f, "alerts (reports)     : %d answered, %d late (after listening stopped)\n",
            b.alerts, b.late_reports);
    fprintf(f, "replies              : %d with nearest nodes, %d with none available\n",
            b.replies_found, b.replies_none);
    fprintf(f, "node events          : %.0f cycles, %.0f local redirects, %.0f reports sent\n",
            tot.cycles, tot.local_redirects, tot.reports);
    fprintf(f, "alerts per node      :");
    for (int i = 0; i < N; i++) fprintf(f, " %d", b.alerts_by_node[i]);
    fprintf(f, "\nmessages             : clock sync %lld, neighbour %.0f, report %lld, reply %d, "
            "termination %d, done %d -> %lld point-to-point; %.0f Allreduce calls\n",
            sync_msgs, tot.nbr_msgs_sent, reports, b.alerts, N, N, p2p_total, tot.allreduce.n);
    fprintf(f, "\ncommunication time per message type (ms)\n");
    fprintf(f, "  %-26s %8s %10s %10s %10s\n", "type", "count", "avg", "min", "max");
    log_acc(f, "neighbour exchange", &tot.xchg);
    log_acc(f, "termination Allreduce", &tot.allreduce);
    log_acc(f, "report (node->base)", &b.report_delay);
    log_acc(f, "reply (base->node)", &tot.reply);
    log_acc(f, "termination (base->node)", &term_delay);
    log_acc(f, "base processing", &b.proc_time);
    fprintf(f, "\nBASE STATION shut down cleanly\n");

    /* one CSV row per message type, appended across runs for the Task 3 graphs */
    FILE *csv = fopen("results.csv", "a");
    if (csv) {
        fseek(csv, 0, SEEK_END);
        if (ftell(csv) == 0)
            fprintf(csv, "topology,m,n,nodes,k,cycle_s,threshold,busy_pct,variant,base_variant,"
                    "period_s,poll_ms,msg_type,count,avg_ms,min_ms,max_ms,alerts,sim_time_s\n");
        csv_row(csv, cfg, "nbr_exchange", &tot.xchg, b.alerts, sim_time);
        csv_row(csv, cfg, "term_allreduce", &tot.allreduce, b.alerts, sim_time);
        csv_row(csv, cfg, "report", &b.report_delay, b.alerts, sim_time);
        csv_row(csv, cfg, "reply", &tot.reply, b.alerts, sim_time);
        csv_row(csv, cfg, "termination", &term_delay, b.alerts, sim_time);
        csv_row(csv, cfg, "base_processing", &b.proc_time, b.alerts, sim_time);
        fclose(csv);
    }

    free(treq);
    free(prep);
    free(preq);
    free(idx);
    free(b.last_report);
    free(b.last_full);
    free(b.alerts_by_node);
    free(b.reply_buf);
    free(b.reply_req);
    if (b.log != stderr) fclose(b.log);
}

/* ======================= main ======================= */
int main(int argc, char **argv) {
    int provided, world_rank, world_size;
    /* only the master thread of each process makes MPI calls */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Config cfg;
    cfg.m            = arg_i(argc, argv, 1, 3);
    cfg.n            = arg_i(argc, argv, 2, 3);
    cfg.k            = arg_i(argc, argv, 3, 6);
    cfg.cycle        = arg_d(argc, argv, 4, 1.0);
    cfg.duration     = arg_d(argc, argv, 5, 30.0);
    cfg.threshold    = arg_i(argc, argv, 6, 2);
    cfg.busy_pct     = arg_i(argc, argv, 7, 60);
    cfg.variant      = arg_i(argc, argv, 8, VAR_NEIGHBOR_ALLGATHER);
    cfg.period       = arg_d(argc, argv, 9, 5.0);
    cfg.base_variant = arg_i(argc, argv, 10, BV_IRECV_TEST);
    cfg.poll_ms      = arg_d(argc, argv, 11, 1.0);

    int bad = 0;
    if (provided < MPI_THREAD_FUNNELED) bad = 1;
    else if (cfg.m < 1 || cfg.n < 1 || cfg.m * cfg.n + 1 != world_size) bad = 2;
    else if (cfg.k < 1 || cfg.k > MAX_PORTS) bad = 3;
    else if (cfg.variant < 0 || cfg.variant > VAR_P2P_ISEND_IRECV ||
             cfg.base_variant < 0 || cfg.base_variant >= NUM_BV) bad = 4;
    else if (cfg.cycle <= 0 || cfg.period < 0 || cfg.poll_ms <= 0) bad = 5;
    if (bad) {
        if (world_rank == 0) {
            if (bad == 1) fprintf(stderr, "MPI_THREAD_FUNNELED not supported\n");
            if (bad == 2) fprintf(stderr, "need -np %d for a %dx%d grid (got %d)\n",
                                  cfg.m * cfg.n + 1, cfg.m, cfg.n, world_size);
            if (bad == 3) fprintf(stderr, "k must be 1..%d\n", MAX_PORTS);
            if (bad == 4) fprintf(stderr, "variant must be 0..2 and base_variant 0..%d\n", NUM_BV - 1);
            if (bad == 5) fprintf(stderr, "cycle and poll_ms must be > 0, period >= 0\n");
        }
        MPI_Finalize();
        return 1;
    }

    /* base station is NOT part of the grid: split it out (star vs mesh) */
    MPI_Comm nodes_comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank == BASE_RANK ? MPI_UNDEFINED : 0,
                   world_rank, &nodes_comm);

    MPI_Datatype report_t = build_report_type();
    MPI_Datatype reply_t  = build_reply_type();

    if (world_rank == BASE_RANK) {
        clock_sync_base(cfg.m * cfg.n);
        base_run(&cfg, report_t, reply_t);           /* nodes_comm == MPI_COMM_NULL */
    } else {
        int dims[2] = { cfg.m, cfg.n }, periods[2] = { 0, 0 };   /* mesh, not torus */
        MPI_Comm cart_comm;
        MPI_Cart_create(nodes_comm, 2, dims, periods, 0 /* no reorder */, &cart_comm);
        double offset = clock_sync_node();
        node_run(&cfg, cart_comm, report_t, reply_t, offset);
        MPI_Comm_free(&cart_comm);                   /* collective over nodes only */
        MPI_Comm_free(&nodes_comm);
    }

    MPI_Type_free(&report_t);
    MPI_Type_free(&reply_t);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
