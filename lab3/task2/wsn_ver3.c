/* FIT3143 Lab 3 - Distributed WSN of EV charging nodes (Open MPI + OpenMP)
 * Charmaine Chee Hing Yi 33520399 | Nazara Glaukofani Lopulisa 35135271
 *
 * compile: mpicc -fopenmp -Wall -o wsn wsn_ver3.c
 * run:     mpirun -np <m*n+1> ./wsn m n [k cycle duration threshold busy_pct variant period
 *                                       base_variant poll_ms nbr_timeout_ms dead_node out_dir]
 *          e.g. mpirun -np 10 ./wsn 3 3 6 1.0 30 2 60
 *          low-availability test : busy_pct = 90
 *          dead-node test        : dead_node = 4  (node 4 stops answering its neighbours)
 *
 *   variant        node neighbour exchange
 *                    0 ON-DEMAND (default): ask neighbours ONLY when congested (TAG_NBR_REQ/RESP)
 *                    1 periodic MPI_Neighbor_allgather  | 2 periodic MPI_Ineighbor_allgather
 *                    3 periodic Isend/Irecv              (periodic = every node, every cycle)
 *   period         base station "recent report" window (s)
 *   base_variant   base station listener
 *                    0 Iprobe+Recv, blocking Send reply (default)
 *                    1 Irecv+Test, Isend reply
 *                    2 persistent Recv_init per node + Testsome, Isend reply
 *   poll_ms        how often idle processes check for messages (ms)
 *   nbr_timeout_ms on-demand: give up waiting for a neighbour after this long
 *   dead_node      cart rank that stops answering its neighbours (-1 = none, on-demand only)
 *   out_dir        results folder: <out_dir>/run_<m>x<n>_v<variant>_b<base>_<time>/ + results.csv
 *
 * world rank 0 = base station; world ranks 1..m*n = charging nodes (cart rank = world rank - 1) */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <mpi.h>
#include <omp.h>

/* ======================= shared definitions ======================= */

#define BASE_RANK       0          /* world rank of the base station           */
#define MAX_PORTS       64
#define MAX_CAND        64         /* max nearest nodes in a redirect reply    */
#define HISTORY_LEN     16         /* circular buffer of availability          */
#define SYNC_ROUNDS     8          /* ping-pongs per node for clock sync       */
#define NEVER           (-1e300)   /* "no report yet" timestamp                */
#define BASE_TIMEOUT_S  5.0        /* node gives up waiting for the base       */

#define NBR_NONE        -1         /* no neighbour in this direction (grid edge) */
#define NBR_NO_RESPONSE -2         /* neighbour did not answer before timeout    */

/* neighbour order = MPI_Neighbor_allgather order on a 2D cart comm */
enum { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3, NUM_DIRS = 4 };
static const char *dir_name[NUM_DIRS] = { "up", "down", "left", "right" };

/* one tag per message type */
enum {
    TAG_REPORT   = 1,   /* node -> base : congestion report              */
    TAG_REPLY    = 2,   /* base -> node : nearest available nodes        */
    TAG_TERM     = 3,   /* base -> node : terminate (payload: send time) */
    TAG_DONE     = 4,   /* node -> base : shut down + node statistics    */
    TAG_NBR      = 5,   /* node <-> node: availability (periodic p2p)    */
    TAG_SYNC     = 6,   /* node <-> base: clock synchronisation          */
    TAG_NBR_REQ  = 7,   /* node -> nbr  : "send me your availability"    */
    TAG_NBR_RESP = 8    /* nbr -> node  : availability                   */
};

/* neighbour-exchange implementations compared in Task 3 */
enum { VAR_ON_DEMAND = 0,
       VAR_PERIODIC_NEIGHBOR_ALLGATHER,
       VAR_PERIODIC_INEIGHBOR_ALLGATHER,
       VAR_PERIODIC_P2P,
       NUM_VARIANTS };
static const char *variant_name[NUM_VARIANTS] = { "on-demand", "periodic Neighbor_allgather",
                                                  "periodic Ineighbor_allgather", "periodic Isend/Irecv" };

/* base station listener implementations compared in Task 3 */
enum { BV_IPROBE_RECV = 0,           /* Iprobe + blocking Recv, blocking Send reply     */
       BV_IRECV_TEST = 1,            /* one Irecv(ANY_SOURCE) + Test, Isend reply       */
       BV_PERSISTENT_TESTSOME = 2,   /* Recv_init per node + Testsome, Isend reply      */
       NUM_BV = 3 };
static const char *bv_name[NUM_BV] = { "Iprobe+Recv/Send", "Irecv+Test/Isend", "Recv_init+Testsome/Isend" };

enum { REPLY_FOUND = 0,              /* nearest available node(s) listed             */
       REPLY_NONE_NEARBY = 1,        /* all neighbours reported within the period    */
       REPLY_NONE_IN_GRID = 2 };     /* every node is known to be congested          */

typedef struct {
    int m, n, k;             /* grid rows, cols, ports per node              */
    int threshold;           /* congested when free ports < threshold        */
    int busy_pct;            /* chance (%) a port is in use each update      */
    int variant;             /* node neighbour-exchange implementation       */
    int base_variant;        /* base station listener implementation         */
    int nbr_timeout_ms;      /* on-demand: wait this long for neighbours     */
    int dead_node;           /* cart rank that stops responding, -1 = none   */
    double cycle;            /* seconds per simulation cycle                 */
    double duration;         /* seconds the base station runs the simulation */
    double period;           /* base: a report is "recent" if younger (s)    */
    double poll_ms;          /* idle polling interval (ms)                   */
    char out_dir[200];
    char run_dir[400];
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
    int nbr_avail[NUM_DIRS];                        /* -1 none, -2 no response */
    int nbr_no_response;                            /* how many timed out      */
    int msgs_with_neighbours;                       /* this detection          */
    int bytes_with_neighbours;
    /* doubles */
    double send_wtime;                              /* base clock at send                   */
    double nbr_comm_time;                           /* neighbour round (s), on this node    */
    double last_base_rtt;                           /* previous alert round trip (s), -1 none */
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
    double cycles, congested, local_redirects, reports, replies;
    double nbr_msgs_sent, nbr_msgs_recv, nbr_bytes, nbr_timeouts, base_timeouts;
    double term_delay;               /* TERM one-way delay (s)                              */
    Acc nbr_round;                   /* one neighbour round / periodic exchange             */
    Acc nbr_rtt;                     /* request -> response per neighbour (on-demand)       */
    Acc alert_rtt;                   /* report -> reply round trip, measured on the node    */
    Acc reply;                       /* base -> node reply one-way delay (synced clocks)    */
    Acc allreduce;                   /* periodic termination-check Allreduce                */
} NodeStats;
#define NODE_STATS_LEN ((int)(sizeof(NodeStats) / sizeof(double)))

static void node_stats_init(NodeStats *s) {
    memset(s, 0, sizeof *s);
    acc_init(&s->nbr_round); acc_init(&s->nbr_rtt); acc_init(&s->alert_rtt);
    acc_init(&s->reply);     acc_init(&s->allreduce);
}

/* ======================= MPI derived datatypes ======================= */
/* Report: block of ints followed by block of doubles */
MPI_Datatype build_report_type(void) {
    int n_ints = (int)((offsetof(Report, send_wtime) - offsetof(Report, cart_rank)) / sizeof(int));
    int blocklens[2] = { n_ints, 3 };
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
 * smallest RTT is kept, so the offset error is at most RTT_min / 2.
 * Round trips (alert -> reply, request -> response) are measured on ONE process
 * and do not depend on this. */
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

/* ---------- reading the shared array ---------- */
/* once per cycle: consistent snapshot + record in history */
static int snapshot_availability(int k, int *states_out) {
    int free_ports = 0;
    omp_set_lock(&state_lock);
    for (int i = 0; i < k; i++) {
        states_out[i] = port_state[i];
        free_ports += !port_state[i];
    }
    history[history_count % HISTORY_LEN] = free_ports;
    history_count++;
    omp_unset_lock(&state_lock);
    return free_ports;
}

/* when a neighbour asks: current number of free ports */
static int count_free(int k) {
    int free_ports = 0;
    omp_set_lock(&state_lock);
    for (int i = 0; i < k; i++) free_ports += !port_state[i];
    omp_unset_lock(&state_lock);
    return free_ports;
}

/* ---------- everything the communication thread needs ---------- */
typedef struct {
    const Config *cfg;
    MPI_Comm cart;
    MPI_Datatype report_t, reply_t;
    FILE *log;
    int cart_rank, coords[2], num_nbrs, dead;
    int nbr[NUM_DIRS], nbr_coords[NUM_DIRS][2];
    /* on-demand neighbour messages: buffers must stay valid until the Isend completes */
    int req_buf[2];                       /* {my cart rank, round id}  */
    MPI_Request req_req[NUM_DIRS];
    int resp_buf[NUM_DIRS][2];            /* {free ports, round id}    */
    MPI_Request resp_req[NUM_DIRS];
    /* base station */
    Report report;
    MPI_Request report_req;
    int waiting_base;
    double t_alert, last_base_rtt;
    NodeStats st;
} Node;

/* one on-demand "ask the neighbours" round */
typedef struct {
    int active, id;
    double t_start;
    int responded[NUM_DIRS], avail[NUM_DIRS];
    int msgs, bytes;
    /* the cycle row is printed when the round finishes, with the neighbours' answers */
    int cycle, my_avail, states[MAX_PORTS];
    struct tm tmv;
} Round;

static int dir_of(const Node *nd, int cart_src) {
    for (int d = 0; d < NUM_DIRS; d++) if (nd->nbr[d] == cart_src) return d;
    return -1;
}

static void print_row(Node *nd, const struct tm *tmv, int avail, const int *states,
                      const int *nbr_avail, double xchg_ms, const char *action) {
    fprintf(nd->log, "%04d %02d %02d %02d %02d %02d | %5d | ", tmv->tm_year + 1900, tmv->tm_mon + 1,
            tmv->tm_mday, tmv->tm_hour, tmv->tm_min, tmv->tm_sec, avail);
    for (int i = 0; i < nd->cfg->k; i++) fputc(states[i] ? '1' : '0', nd->log);
    if (nbr_avail)
        fprintf(nd->log, " | %2d %2d %2d %2d | %8.3f | %s\n",
                nbr_avail[UP], nbr_avail[DOWN], nbr_avail[LEFT], nbr_avail[RIGHT], xchg_ms, action);
    else
        fprintf(nd->log, " |  -  -  -  - |        - | %s\n", action);
}

/* ---------- report to the base station: ONE message, derived datatype ---------- */
static void send_report(Node *nd, int cycle, const struct tm *tmv, int avail, const int *states,
                        const int *nbr_avail, int msgs, int bytes, double xchg) {
    const Config *cfg = nd->cfg;
    MPI_Wait(&nd->report_req, MPI_STATUS_IGNORE);          /* report buffer reusable */
    Report *r = &nd->report;
    memset(r, 0, sizeof *r);
    r->cart_rank = nd->cart_rank;
    r->world_rank = nd->cart_rank + 1;
    r->coords[0] = nd->coords[0]; r->coords[1] = nd->coords[1];
    r->cycle = cycle;
    r->year = tmv->tm_year + 1900; r->month = tmv->tm_mon + 1; r->day = tmv->tm_mday;
    r->hour = tmv->tm_hour; r->minute = tmv->tm_min; r->second = tmv->tm_sec;
    r->k = cfg->k;
    r->availability = avail;
    memcpy(r->port_state, states, sizeof(int) * cfg->k);
    omp_set_lock(&state_lock);
    for (int i = 0; i < HISTORY_LEN; i++) {                /* oldest -> newest, -1 = unused */
        int idx = history_count - HISTORY_LEN + i;
        r->history[i] = idx >= 0 ? history[idx % HISTORY_LEN] : -1;
    }
    omp_unset_lock(&state_lock);
    r->num_neighbours = nd->num_nbrs;
    for (int d = 0; d < NUM_DIRS; d++) {
        r->nbr_rank[d] = nd->nbr[d] == MPI_PROC_NULL ? -1 : nd->nbr[d];
        r->nbr_coords[d][0] = nd->nbr_coords[d][0];
        r->nbr_coords[d][1] = nd->nbr_coords[d][1];
        r->nbr_avail[d] = nbr_avail[d];
        if (nbr_avail[d] == NBR_NO_RESPONSE) r->nbr_no_response++;
    }
    r->msgs_with_neighbours = msgs;
    r->bytes_with_neighbours = bytes;
    r->nbr_comm_time = xchg;
    r->last_base_rtt = nd->last_base_rtt;

    nd->t_alert = MPI_Wtime();                             /* round trip measured on THIS process */
    r->send_wtime = base_clock();                          /* one-way delay measured at the base  */
    MPI_Isend(r, 1, nd->report_t, BASE_RANK, TAG_REPORT, MPI_COMM_WORLD, &nd->report_req);
    nd->waiting_base = 1;
    nd->st.reports++;
}

/* best neighbour that is NOT below the threshold, -1 if none */
static int best_neighbour(const Node *nd, const int *nbr_avail) {
    int best = -1;
    for (int d = 0; d < NUM_DIRS; d++)
        if (nbr_avail[d] >= nd->cfg->threshold && (best < 0 || nbr_avail[d] > nbr_avail[best]))
            best = d;
    return best;
}

/* ---------- on-demand: answer requests, collect responses (never blocks) ---------- */
static void service_neighbours(Node *nd, Round *rd, int answer) {
    for (;;) {
        int flag = 0, msg[2];
        MPI_Status s;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, nd->cart, &flag, &s);
        if (!flag) return;                                  /* nothing waiting */
        int d = dir_of(nd, s.MPI_SOURCE);

        if (s.MPI_TAG == TAG_NBR_REQ) {
            MPI_Recv(msg, 2, MPI_INT, s.MPI_SOURCE, TAG_NBR_REQ, nd->cart, MPI_STATUS_IGNORE);
            nd->st.nbr_msgs_recv++; nd->st.nbr_bytes += sizeof msg;
            if (answer && d >= 0) {
                MPI_Wait(&nd->resp_req[d], MPI_STATUS_IGNORE);   /* previous answer delivered */
                nd->resp_buf[d][0] = count_free(nd->cfg->k);
                nd->resp_buf[d][1] = msg[1];                     /* echo the round id */
                MPI_Isend(nd->resp_buf[d], 2, MPI_INT, s.MPI_SOURCE, TAG_NBR_RESP, nd->cart, &nd->resp_req[d]);
                nd->st.nbr_msgs_sent++; nd->st.nbr_bytes += sizeof msg;
            }
        } else if (s.MPI_TAG == TAG_NBR_RESP) {
            MPI_Recv(msg, 2, MPI_INT, s.MPI_SOURCE, TAG_NBR_RESP, nd->cart, MPI_STATUS_IGNORE);
            nd->st.nbr_msgs_recv++; nd->st.nbr_bytes += sizeof msg;
            /* ignore late answers that belong to an older round */
            if (rd && rd->active && d >= 0 && msg[1] == rd->id && !rd->responded[d]) {
                rd->responded[d] = 1;
                rd->avail[d] = msg[0];
                rd->msgs++; rd->bytes += sizeof msg;
                acc_add(&nd->st.nbr_rtt, MPI_Wtime() - rd->t_start);
            }
        } else {                                            /* unexpected: consume it */
            int cnt, junk[16];
            MPI_Get_count(&s, MPI_INT, &cnt);
            if (cnt > 16) cnt = 16;
            MPI_Recv(junk, cnt, MPI_INT, s.MPI_SOURCE, s.MPI_TAG, nd->cart, MPI_STATUS_IGNORE);
            fprintf(nd->log, "  WARNING unexpected tag %d from node %d\n", s.MPI_TAG, s.MPI_SOURCE);
        }
    }
}

static void start_round(Node *nd, Round *rd, int cycle, int avail, const int *states, const struct tm *tmv) {
    rd->active = 1;
    rd->id++;
    rd->t_start = MPI_Wtime();
    rd->msgs = rd->bytes = 0;
    rd->cycle = cycle; rd->my_avail = avail; rd->tmv = *tmv;
    memcpy(rd->states, states, sizeof(int) * nd->cfg->k);

    MPI_Waitall(NUM_DIRS, nd->req_req, MPI_STATUSES_IGNORE);   /* previous requests delivered */
    nd->req_buf[0] = nd->cart_rank;
    nd->req_buf[1] = rd->id;
    for (int d = 0; d < NUM_DIRS; d++) {
        rd->responded[d] = 0;
        rd->avail[d] = nd->nbr[d] == MPI_PROC_NULL ? NBR_NONE : NBR_NO_RESPONSE;
        if (nd->nbr[d] == MPI_PROC_NULL) continue;
        MPI_Isend(nd->req_buf, 2, MPI_INT, nd->nbr[d], TAG_NBR_REQ, nd->cart, &nd->req_req[d]);
        rd->msgs++; rd->bytes += sizeof nd->req_buf;
        nd->st.nbr_msgs_sent++; nd->st.nbr_bytes += sizeof nd->req_buf;
    }
}

static int round_complete(const Node *nd, const Round *rd) {
    int got = 0;
    for (int d = 0; d < NUM_DIRS; d++) got += rd->responded[d];
    return got == nd->num_nbrs || (MPI_Wtime() - rd->t_start) * 1e3 > nd->cfg->nbr_timeout_ms;
}

static void finish_round(Node *nd, Round *rd, int terminating) {
    rd->active = 0;
    double xchg = MPI_Wtime() - rd->t_start;
    acc_add(&nd->st.nbr_round, xchg);

    for (int d = 0; d < NUM_DIRS; d++)
        if (rd->avail[d] == NBR_NO_RESPONSE) {
            nd->st.nbr_timeouts++;
            fprintf(nd->log, "  WARNING neighbour %s node %d (%d,%d) did not respond within %d ms - possibly dead\n",
                    dir_name[d], nd->nbr[d], nd->nbr_coords[d][0], nd->nbr_coords[d][1], nd->cfg->nbr_timeout_ms);
        }

    char action[128];
    int best = best_neighbour(nd, rd->avail);
    if (best >= 0) {
        /* a neighbour has room: redirect locally, no base station traffic */
        nd->st.local_redirects++;
        snprintf(action, sizeof action, "CONGESTED -> redirect EVs to node %d (%d,%d) avail=%d",
                 nd->nbr[best], nd->nbr_coords[best][0], nd->nbr_coords[best][1], rd->avail[best]);
    } else if (terminating) {
        snprintf(action, sizeof action, "CONGESTED, neighbours full (shutting down: no alert)");
    } else {
        /* node and all (responding) neighbours congested: alert the base station */
        send_report(nd, rd->cycle, &rd->tmv, rd->my_avail, rd->states, rd->avail, rd->msgs, rd->bytes, xchg);
        snprintf(action, sizeof action, "CONGESTED, all neighbours full -> ALERT base station");
    }
    print_row(nd, &rd->tmv, rd->my_avail, rd->states, rd->avail, xchg * 1e3, action);
}

/* ---------- periodic variants: every node exchanges every cycle ---------- */
static void periodic_exchange(Node *nd, int my_avail, int nbr_avail[NUM_DIRS]) {
    for (int d = 0; d < NUM_DIRS; d++) nbr_avail[d] = NBR_NONE;   /* PROC_NULL slots stay -1 */

    if (nd->cfg->variant == VAR_PERIODIC_NEIGHBOR_ALLGATHER) {
        MPI_Neighbor_allgather(&my_avail, 1, MPI_INT, nbr_avail, 1, MPI_INT, nd->cart);
    } else if (nd->cfg->variant == VAR_PERIODIC_INEIGHBOR_ALLGATHER) {
        MPI_Request req;
        MPI_Ineighbor_allgather(&my_avail, 1, MPI_INT, nbr_avail, 1, MPI_INT, nd->cart, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else {   /* VAR_PERIODIC_P2P: sends/recvs to MPI_PROC_NULL are no-ops */
        MPI_Request reqs[2 * NUM_DIRS];
        for (int d = 0; d < NUM_DIRS; d++) {
            MPI_Irecv(&nbr_avail[d], 1, MPI_INT, nd->nbr[d], TAG_NBR, nd->cart, &reqs[d]);
            MPI_Isend(&my_avail, 1, MPI_INT, nd->nbr[d], TAG_NBR, nd->cart, &reqs[NUM_DIRS + d]);
        }
        MPI_Waitall(2 * NUM_DIRS, reqs, MPI_STATUSES_IGNORE);
    }
    nd->st.nbr_msgs_sent += nd->num_nbrs;
    nd->st.nbr_msgs_recv += nd->num_nbrs;
    nd->st.nbr_bytes     += 2.0 * nd->num_nbrs * sizeof(int);
}

/* ---------- message from the base station ---------- */
static void log_reply(FILE *log, const Reply *r, double rtt, double oneway) {
    if (r->status == REPLY_FOUND) {
        fprintf(log, "  [base reply] forward EVs to %d node(s) at distance %d:", r->found, r->distance);
        for (int i = 0; i < r->count; i++)
            fprintf(log, " node%d(%d,%d)", r->rank[i], r->coords[i][0], r->coords[i][1]);
    } else if (r->status == REPLY_NONE_NEARBY) {
        fprintf(log, "  [base reply] no nodes available nearby");
    } else {
        fprintf(log, "  [base reply] no available node in the whole grid");
    }
    fprintf(log, " (round trip %.3f ms, base->node %.3f ms)\n", rtt * 1e3, oneway * 1e3);
}

static void handle_base_message(Node *nd, MPI_Status *s, int *term) {
    if (s->MPI_TAG == TAG_REPLY) {
        Reply reply;
        MPI_Recv(&reply, 1, nd->reply_t, BASE_RANK, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double rtt = MPI_Wtime() - nd->t_alert;              /* both times taken on this node   */
        double oneway = base_clock() - reply.send_wtime;     /* needs the synchronised clock    */
        nd->last_base_rtt = rtt;
        nd->waiting_base = 0;
        nd->st.replies++;
        acc_add(&nd->st.alert_rtt, rtt);
        acc_add(&nd->st.reply, oneway);
        log_reply(nd->log, &reply, rtt, oneway);
    } else if (s->MPI_TAG == TAG_TERM) {
        double term_send;
        MPI_Recv(&term_send, 1, MPI_DOUBLE, BASE_RANK, TAG_TERM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        nd->st.term_delay = base_clock() - term_send;
        *term = 1;
    } else {
        int cnt, junk[16];
        MPI_Get_count(s, MPI_INT, &cnt);
        if (cnt > 16) cnt = 16;
        MPI_Recv(junk, cnt, MPI_INT, BASE_RANK, s->MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

/* ---------- communication thread (master, the only thread calling MPI) ---------- */
static void comm_thread(Node *nd) {
    const Config *cfg = nd->cfg;
    double poll = cfg->poll_ms / 1e3;
    MPI_Comm_rank(nd->cart, &nd->cart_rank);
    MPI_Cart_coords(nd->cart, nd->cart_rank, 2, nd->coords);
    MPI_Cart_shift(nd->cart, 0, 1, &nd->nbr[UP],   &nd->nbr[DOWN]);
    MPI_Cart_shift(nd->cart, 1, 1, &nd->nbr[LEFT], &nd->nbr[RIGHT]);
    nd->num_nbrs = 0;
    for (int d = 0; d < NUM_DIRS; d++) {
        nd->nbr_coords[d][0] = nd->nbr_coords[d][1] = -1;
        if (nd->nbr[d] != MPI_PROC_NULL) { MPI_Cart_coords(nd->cart, nd->nbr[d], 2, nd->nbr_coords[d]); nd->num_nbrs++; }
        nd->req_req[d] = nd->resp_req[d] = MPI_REQUEST_NULL;
    }
    nd->report_req = MPI_REQUEST_NULL;
    nd->last_base_rtt = -1;
    nd->dead = (nd->cart_rank == cfg->dead_node);
    node_stats_init(&nd->st);
    int on_demand = (cfg->variant == VAR_ON_DEMAND);

    fprintf(nd->log, "node cart=%d world=%d coords=(%d,%d) k=%d threshold=%d variant=%d (%s) clock_offset=%.6f s%s\n",
            nd->cart_rank, nd->cart_rank + 1, nd->coords[0], nd->coords[1], cfg->k, cfg->threshold,
            cfg->variant, variant_name[cfg->variant], g_clock_offset,
            nd->dead ? "  ** SIMULATED DEAD NODE **" : "");
    for (int d = 0; d < NUM_DIRS; d++) {
        if (nd->nbr[d] == MPI_PROC_NULL) fprintf(nd->log, "  neighbour %-5s : none\n", dir_name[d]);
        else fprintf(nd->log, "  neighbour %-5s : node %d (%d,%d)\n", dir_name[d], nd->nbr[d],
                     nd->nbr_coords[d][0], nd->nbr_coords[d][1]);
    }
    fprintf(nd->log, "YYYY MM DD HH MM SS | avail | ports(1=used) | nbr avail u/d/l/r (-1 none, -2 no reply) | xchg(ms) | action\n");

    Round rd;
    memset(&rd, 0, sizeof rd);
    int states[MAX_PORTS], nbr_avail[NUM_DIRS], term = 0, cycle = 0;
    double next_cycle = MPI_Wtime() + cfg->cycle;

    for (;;) {
        /* 1. base station: Iprobe first, Recv only if something is there.
         *    Messages from the base arrive in order, so a REPLY is always seen before a TERM. */
        int flag = 0;
        MPI_Status s;
        MPI_Iprobe(BASE_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &s);
        if (flag) handle_base_message(nd, &s, &term);

        /* 2. neighbours (on-demand): answer requests at any time, collect answers */
        if (on_demand) {
            service_neighbours(nd, &rd, !nd->dead);
            if (rd.active && round_complete(nd, &rd)) finish_round(nd, &rd, term);
        }

        /* 3. base station never answered (dead / overloaded) */
        if (nd->waiting_base && MPI_Wtime() - nd->t_alert > BASE_TIMEOUT_S) {
            nd->waiting_base = 0;
            nd->st.base_timeouts++;
            fprintf(nd->log, "  WARNING no reply from base station within %.0f s\n", BASE_TIMEOUT_S);
        }

        /* 4. once per cycle: read the ports and decide */
        double now = MPI_Wtime();
        if (now >= next_cycle && !(on_demand && term)) {
            next_cycle += cfg->cycle;
            if (next_cycle < now) next_cycle = now + cfg->cycle;

            if (!on_demand) {
                /* periodic: neighbours must call the collective the same number of times,
                 * so every node agrees to stop in the same cycle */
                int stop = 0;
                double t0 = MPI_Wtime();
                MPI_Allreduce(&term, &stop, 1, MPI_INT, MPI_MAX, nd->cart);
                acc_add(&nd->st.allreduce, MPI_Wtime() - t0);
                if (stop) break;
            }

            cycle++;
            nd->st.cycles++;
            int avail = snapshot_availability(cfg->k, states);
            time_t t = time(NULL);
            struct tm tmv;
            localtime_r(&t, &tmv);
            int congested = avail < cfg->threshold;
            if (congested) nd->st.congested++;

            if (on_demand) {
                if (!congested)            print_row(nd, &tmv, avail, states, NULL, 0, "ok");
                else if (nd->dead)         print_row(nd, &tmv, avail, states, NULL, 0, "CONGESTED (dead node: silent)");
                else if (rd.active)        print_row(nd, &tmv, avail, states, NULL, 0, "CONGESTED, previous neighbour round still open");
                else if (nd->waiting_base) print_row(nd, &tmv, avail, states, NULL, 0, "CONGESTED, awaiting base reply");
                else start_round(nd, &rd, cycle, avail, states, &tmv);   /* row printed when it finishes */
            } else {
                double t0 = MPI_Wtime();
                periodic_exchange(nd, avail, nbr_avail);
                double xchg = MPI_Wtime() - t0;
                acc_add(&nd->st.nbr_round, xchg);

                char action[128] = "ok";
                if (congested) {
                    int best = best_neighbour(nd, nbr_avail);
                    if (best >= 0) {
                        nd->st.local_redirects++;
                        snprintf(action, sizeof action, "CONGESTED -> redirect EVs to node %d (%d,%d) avail=%d",
                                 nd->nbr[best], nd->nbr_coords[best][0], nd->nbr_coords[best][1], nbr_avail[best]);
                    } else if (nd->waiting_base) {
                        snprintf(action, sizeof action, "CONGESTED, neighbours full, awaiting base reply");
                    } else {
                        send_report(nd, cycle, &tmv, avail, states, nbr_avail,
                                    2 * nd->num_nbrs, 2 * nd->num_nbrs * (int)sizeof(int), xchg);
                        snprintf(action, sizeof action, "CONGESTED, all neighbours full -> ALERT base station");
                    }
                }
                print_row(nd, &tmv, avail, states, nbr_avail, xchg * 1e3, action);
            }
        }

        /* 5. on-demand: leave once terminated and our last round is finished */
        if (on_demand && term && !rd.active) break;
        sleep_seconds(poll);
    }

    /* ---------- graceful shutdown of communication ---------- */
    #pragma omp atomic write
    g_terminate = 1;                                   /* port threads exit their loops */

    if (on_demand) {
        /* non-blocking barrier: keep answering neighbours until EVERY node has finished its
         * last round, so nobody is left waiting for an answer that never comes */
        MPI_Request bar;
        int done = 0;
        MPI_Ibarrier(nd->cart, &bar);
        while (!done) {
            service_neighbours(nd, NULL, !nd->dead);
            MPI_Test(&bar, &done, MPI_STATUS_IGNORE);
            if (!done) sleep_seconds(poll);
        }
        for (double end = MPI_Wtime() + 0.02; MPI_Wtime() < end; sleep_seconds(poll))
            service_neighbours(nd, NULL, 0);                /* drain anything still in flight */
        MPI_Waitall(NUM_DIRS, nd->req_req, MPI_STATUSES_IGNORE);
        MPI_Waitall(NUM_DIRS, nd->resp_req, MPI_STATUSES_IGNORE);
    }
    MPI_Wait(&nd->report_req, MPI_STATUS_IGNORE);
    while (!term) {                                    /* periodic: stopped because a neighbour got TERM */
        MPI_Status s;
        MPI_Probe(BASE_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &s);
        handle_base_message(nd, &s, &term);
    }
    /* tell the base station we are done (and hand over our statistics) */
    MPI_Send(&nd->st, NODE_STATS_LEN, MPI_DOUBLE, BASE_RANK, TAG_DONE, MPI_COMM_WORLD);

    const NodeStats *st = &nd->st;
    fprintf(nd->log, "\nTERMINATED by base station (termination delay %.3f ms)\n", st->term_delay * 1e3);
    fprintf(nd->log, "summary: cycles=%.0f congested=%.0f local_redirects=%.0f reports_to_base=%.0f replies=%.0f\n",
            st->cycles, st->congested, st->local_redirects, st->reports, st->replies);
    fprintf(nd->log, "         neighbour msgs sent=%.0f recv=%.0f bytes=%.0f rounds=%.0f avg_round=%.3f ms timeouts=%.0f\n",
            st->nbr_msgs_sent, st->nbr_msgs_recv, st->nbr_bytes, st->nbr_round.n,
            st->nbr_round.n ? st->nbr_round.sum / st->nbr_round.n * 1e3 : 0.0, st->nbr_timeouts);
    fprintf(nd->log, "         alert round trip avg=%.3f ms\n",
            st->alert_rtt.n ? st->alert_rtt.sum / st->alert_rtt.n * 1e3 : 0.0);
}

void node_run(const Config *cfg, MPI_Comm cart, MPI_Datatype report_t, MPI_Datatype reply_t,
              double clock_offset) {
    static Node nd;                                    /* large (holds a Report): keep off the stack */
    memset(&nd, 0, sizeof nd);
    nd.cfg = cfg; nd.cart = cart; nd.report_t = report_t; nd.reply_t = reply_t;
    MPI_Comm_rank(cart, &nd.cart_rank);

    char fname[480];
    snprintf(fname, sizeof fname, "%s/node_%d.log", cfg->run_dir, nd.cart_rank);
    nd.log = fopen(fname, "w");
    if (!nd.log) nd.log = stderr;

    omp_init_lock(&state_lock);
    memset(port_state, 0, sizeof port_state);
    g_terminate = 0;
    g_clock_offset = clock_offset;

    omp_set_dynamic(0);
    #pragma omp parallel num_threads(cfg->k + 1)
    {
        int tid = omp_get_thread_num();
        if (tid == 0) comm_thread(&nd);                /* master = FUNNELED */
        else          port_thread(cfg, tid - 1, nd.cart_rank);
    }   /* implicit barrier: every port thread has returned */

    omp_destroy_lock(&state_lock);
    if (nd.log != stderr) fclose(nd.log);
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
 *   last_silent[j] - when a neighbour of j reported that j did NOT respond
 *                    (possibly dead): never suggest it to EVs while that is recent
 * On a report from node X:
 *   - every neighbour of X reported within `period`  -> "no nodes available nearby"
 *   - otherwise return ALL nodes at the smallest grid distance from X that are
 *     not known full (or silent) within `period` (ring 2 = neighbours of neighbours
 *     first, expanding outwards if that ring is congested too). */
typedef struct {
    const Config *cfg;
    int           N;
    FILE         *log;
    MPI_Datatype  reply_t;
    double        start;
    double       *last_report;
    double       *last_full;
    double       *last_silent;
    int          *alerts_by_node;
    int          *silent_by_node;
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

    /* nearest = smallest Manhattan distance (hops in the mesh) among nodes not known full/silent */
    int best = INT_MAX;
    for (int j = 0; j < b->N; j++) {
        int dist = abs(j / n - r0) + abs(j % n - c0);
        if (dist > 0 && dist < best && now - b->last_full[j] > cfg->period
            && now - b->last_silent[j] > cfg->period) best = dist;
    }
    if (best == INT_MAX) {
        r->status = REPLY_NONE_IN_GRID;
        return recent;
    }

    r->status = REPLY_FOUND;
    r->distance = best;
    for (int j = 0; j < b->N; j++) {   /* report ALL nodes tied at the nearest distance */
        int dist = abs(j / n - r0) + abs(j % n - c0);
        if (dist != best || now - b->last_full[j] <= cfg->period || now - b->last_silent[j] <= cfg->period)
            continue;
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
        fprintf(f, " %s=node%d(%d,%d) ", dir_name[d], rep->nbr_rank[d],
                rep->nbr_coords[d][0], rep->nbr_coords[d][1]);
        if (rep->nbr_avail[d] == NBR_NO_RESPONSE) fprintf(f, "NO-RESPONSE(possibly dead)");
        else fprintf(f, "avail=%d", rep->nbr_avail[d]);
        if (nbr_age[d] > 1e200) fprintf(f, " last_report=never");
        else fprintf(f, " last_report=%.2fs ago", nbr_age[d]);
    }
    fputc('\n', f);

    fprintf(f, "    neighbour communication: %d msgs, %d bytes, %.3f ms | report delay %.3f ms",
            rep->msgs_with_neighbours, rep->bytes_with_neighbours, rep->nbr_comm_time * 1e3, delay * 1e3);
    if (rep->last_base_rtt >= 0) fprintf(f, " | node's previous alert round trip %.3f ms", rep->last_base_rtt * 1e3);
    fputc('\n', f);

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

    /* this report proves the node (alive) and its reported neighbours are full now */
    b->last_report[me] = b->last_full[me] = recv_time;
    b->last_silent[me] = NEVER;
    for (int d = 0; d < NUM_DIRS; d++) {
        int j = rep->nbr_rank[d];
        if (j < 0) continue;
        if (rep->nbr_avail[d] >= 0 && rep->nbr_avail[d] < cfg->threshold) b->last_full[j] = recv_time;
        if (rep->nbr_avail[d] == NBR_NO_RESPONSE) {      /* neighbour tells us a node may be dead */
            if (recv_time - b->last_silent[j] > cfg->period)
                fprintf(b->log, "[t=%9.3fs] NOTICE node %d (%d,%d) reported unresponsive by node %d - "
                        "not suggested to EVs while silent\n", recv_time - b->start, j, j / cfg->n, j % cfg->n, me);
            b->last_silent[j] = recv_time;
            b->silent_by_node[j]++;
        }
    }

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
    if (a->n == 0) fprintf(f, "  %-30s %8d %10s %10s %10s\n", name, 0, "-", "-", "-");
    else fprintf(f, "  %-30s %8.0f %10.3f %10.3f %10.3f\n", name, a->n,
                 a->sum / a->n * 1e3, a->min * 1e3, a->max * 1e3);
}

static void csv_row(FILE *f, const Config *cfg, const char *type, const Acc *a, const Base *b,
                    const NodeStats *tot, double sim_time) {
    fprintf(f, "%s,%d,%d,%d,%d,%.3f,%d,%d,%d,%d,%.3f,%.3f,%d,%s,%.0f,",
            (cfg->m == 1 || cfg->n == 1) ? "line" : "grid", cfg->m, cfg->n, cfg->m * cfg->n, cfg->k,
            cfg->cycle, cfg->threshold, cfg->busy_pct, cfg->variant, cfg->base_variant,
            cfg->period, cfg->poll_ms, cfg->dead_node, type, a->n);
    if (a->n > 0) fprintf(f, "%.6f,%.6f,%.6f", a->sum / a->n * 1e3, a->min * 1e3, a->max * 1e3);
    else fprintf(f, ",,");
    fprintf(f, ",%d,%.0f,%.0f,%.0f,%.0f,%.3f\n", b->alerts, tot->congested, tot->local_redirects,
            tot->nbr_msgs_sent, tot->nbr_bytes, sim_time);
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
    b.last_silent    = malloc(N * sizeof(double));
    b.alerts_by_node = calloc(N, sizeof(int));
    b.silent_by_node = calloc(N, sizeof(int));
    b.reply_buf      = calloc(N, sizeof(Reply));
    b.reply_req      = malloc(N * sizeof(MPI_Request));
    if (!b.last_report || !b.last_full || !b.last_silent || !b.alerts_by_node || !b.silent_by_node ||
        !b.reply_buf || !b.reply_req) {
        fprintf(stderr, "base station: out of memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < N; i++) {
        b.last_report[i] = b.last_full[i] = b.last_silent[i] = NEVER;
        b.reply_req[i] = MPI_REQUEST_NULL;
    }
    acc_init(&b.report_delay);
    acc_init(&b.proc_time);

    char fname[480];
    snprintf(fname, sizeof fname, "%s/base_station.log", cfg->run_dir);
    b.log = fopen(fname, "w");
    if (!b.log) b.log = stderr;

    time_t now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    int report_bytes, reply_bytes;
    MPI_Type_size(report_t, &report_bytes);
    MPI_Type_size(reply_t, &reply_bytes);
    fprintf(b.log, "BASE STATION started %04d-%02d-%02d %02d:%02d:%02d\n",
            tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    fprintf(b.log, "grid=%dx%d (%d nodes, %s) k=%d threshold=%d busy=%d%% cycle=%.2fs duration=%.1fs\n",
            cfg->m, cfg->n, N, (cfg->m == 1 || cfg->n == 1) ? "line" : "grid", cfg->k, cfg->threshold,
            cfg->busy_pct, cfg->cycle, cfg->duration);
    fprintf(b.log, "node exchange=%d (%s), base listener=%d (%s), period=%.2fs, poll=%.2fms, "
            "nbr_timeout=%dms, dead_node=%d\n", cfg->variant, variant_name[cfg->variant], cfg->base_variant,
            bv_name[cfg->base_variant], cfg->period, cfg->poll_ms, cfg->nbr_timeout_ms, cfg->dead_node);
    fprintf(b.log, "message sizes: report %d B, reply %d B, termination %d B, neighbour request/response %d B\n\n",
            report_bytes, reply_bytes, (int)sizeof(double), (int)(2 * sizeof(int)));
    printf("Base station: %dx%d grid, %d nodes, %s, listener %s, running %.0f s\n",
           cfg->m, cfg->n, N, variant_name[cfg->variant], bv_name[cfg->base_variant], cfg->duration);
    fflush(stdout);

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
        if (cfg->base_variant == BV_IPROBE_RECV) {
            /* Iprobe: is a report waiting? Only then call Recv */
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_REPORT, MPI_COMM_WORLD, &got, &st);
            if (got) {
                MPI_Recv(&rep, 1, report_t, st.MPI_SOURCE, TAG_REPORT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                handle_report(&b, &rep, MPI_Wtime());
            }
        } else if (cfg->base_variant == BV_IRECV_TEST) {
            MPI_Test(&req, &got, MPI_STATUS_IGNORE);
            if (got) {
                handle_report(&b, &rep, MPI_Wtime());
                MPI_Irecv(&rep, 1, report_t, MPI_ANY_SOURCE, TAG_REPORT, MPI_COMM_WORLD, &req);
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
    printf("Base station: time up, sending TERMINATE to %d nodes\n", N);
    fflush(stdout);
    double term_send = MPI_Wtime();
    MPI_Request *treq = malloc(N * sizeof(MPI_Request));
    if (!treq) MPI_Abort(MPI_COMM_WORLD, 1);
    for (int i = 0; i < N; i++)
        MPI_Isend(&term_send, 1, MPI_DOUBLE, i + 1, TAG_TERM, MPI_COMM_WORLD, &treq[i]);
    MPI_Waitall(N, treq, MPI_STATUSES_IGNORE);

    /* ---------- drain in-flight reports until every node confirms shutdown ---------- */
    NodeStats ns, tot;
    node_stats_init(&tot);
    Acc term_delay;
    acc_init(&term_delay);
    int done = 0;
    while (done < N) {
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
        if (!flag) { sleep_seconds(poll); continue; }
        if (st.MPI_TAG == TAG_DONE) {
            MPI_Recv(&ns, NODE_STATS_LEN, MPI_DOUBLE, st.MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            tot.cycles += ns.cycles;                 tot.congested += ns.congested;
            tot.reports += ns.reports;               tot.replies += ns.replies;
            tot.local_redirects += ns.local_redirects;
            tot.nbr_msgs_sent += ns.nbr_msgs_sent;   tot.nbr_msgs_recv += ns.nbr_msgs_recv;
            tot.nbr_bytes += ns.nbr_bytes;           tot.nbr_timeouts += ns.nbr_timeouts;
            tot.base_timeouts += ns.base_timeouts;
            acc_add(&term_delay, ns.term_delay);
            acc_merge(&tot.nbr_round, &ns.nbr_round);
            acc_merge(&tot.nbr_rtt, &ns.nbr_rtt);
            acc_merge(&tot.alert_rtt, &ns.alert_rtt);
            acc_merge(&tot.reply, &ns.reply);
            acc_merge(&tot.allreduce, &ns.allreduce);
            done++;
        } else if (st.MPI_TAG == TAG_REPORT) {
            MPI_Recv(&rep, 1, report_t, st.MPI_SOURCE, TAG_REPORT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            log_late(&b, &rep);
        } else {
            char junk[64];
            MPI_Recv(junk, sizeof junk, MPI_BYTE, st.MPI_SOURCE, st.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    MPI_Waitall(N, b.reply_req, MPI_STATUSES_IGNORE);   /* every Isend reply has completed */
    double sim_time = MPI_Wtime() - b.start;

    /* ---------- key performance metrics ---------- */
    long long sync_msgs = 2LL * SYNC_ROUNDS * N;
    long long reports = b.alerts + b.late_reports;
    long long base_msgs = reports + b.alerts + 2LL * N;     /* reports + replies + TERM + DONE */
    FILE *f = b.log;
    fprintf(f, "\n================ SUMMARY ================\n");
    fprintf(f, "simulation time      : %.3f s (listening %.3f s, shutdown %.3f s)\n",
            sim_time, listen_end - b.start, sim_time - (listen_end - b.start));
    fprintf(f, "alerts (reports)     : %d answered, %d late (after listening stopped)\n",
            b.alerts, b.late_reports);
    fprintf(f, "replies              : %d with nearest nodes, %d with none available\n",
            b.replies_found, b.replies_none);
    fprintf(f, "node events          : %.0f cycles, %.0f congested, %.0f local redirects, %.0f reports sent\n",
            tot.cycles, tot.congested, tot.local_redirects, tot.reports);
    fprintf(f, "alerts per node      :");
    for (int i = 0; i < N; i++) fprintf(f, " %d", b.alerts_by_node[i]);
    fprintf(f, "\nreported unresponsive:");
    for (int i = 0; i < N; i++) fprintf(f, " %d", b.silent_by_node[i]);
    fprintf(f, "\nneighbour messages   : sent=%.0f recv=%.0f bytes=%.0f timeouts=%.0f\n",
            tot.nbr_msgs_sent, tot.nbr_msgs_recv, tot.nbr_bytes, tot.nbr_timeouts);
    fprintf(f, "messages with base   : %lld (report %lld, reply %d, termination %d, done %d) + clock sync %lld\n",
            base_msgs, reports, b.alerts, N, N, sync_msgs);
    fprintf(f, "\ncommunication time per message type (ms)\n");
    fprintf(f, "  %-30s %8s %10s %10s %10s\n", "type", "count", "avg", "min", "max");
    log_acc(f, "neighbour round (whole)", &tot.nbr_round);
    log_acc(f, "neighbour request->response", &tot.nbr_rtt);
    log_acc(f, "alert->reply round trip", &tot.alert_rtt);
    log_acc(f, "report (node->base, one-way)", &b.report_delay);
    log_acc(f, "reply (base->node, one-way)", &tot.reply);
    log_acc(f, "termination (base->node)", &term_delay);
    log_acc(f, "base processing", &b.proc_time);
    log_acc(f, "termination Allreduce", &tot.allreduce);
    fprintf(f, "\nBASE STATION shut down cleanly\n");

    /* one CSV row per message type, appended across runs for the Task 3 graphs */
    char csv_path[300];
    snprintf(csv_path, sizeof csv_path, "%s/results.csv", cfg->out_dir);
    FILE *csv = fopen(csv_path, "a");
    if (csv) {
        fseek(csv, 0, SEEK_END);
        if (ftell(csv) == 0)
            fprintf(csv, "topology,m,n,nodes,k,cycle_s,threshold,busy_pct,variant,base_variant,"
                    "period_s,poll_ms,dead_node,msg_type,count,avg_ms,min_ms,max_ms,"
                    "alerts,congested,local_redirects,nbr_msgs,nbr_bytes,sim_time_s\n");
        csv_row(csv, cfg, "nbr_round",       &tot.nbr_round,  &b, &tot, sim_time);
        csv_row(csv, cfg, "nbr_rtt",         &tot.nbr_rtt,    &b, &tot, sim_time);
        csv_row(csv, cfg, "alert_rtt",       &tot.alert_rtt,  &b, &tot, sim_time);
        csv_row(csv, cfg, "report",          &b.report_delay, &b, &tot, sim_time);
        csv_row(csv, cfg, "reply",           &tot.reply,      &b, &tot, sim_time);
        csv_row(csv, cfg, "termination",     &term_delay,     &b, &tot, sim_time);
        csv_row(csv, cfg, "base_processing", &b.proc_time,    &b, &tot, sim_time);
        csv_row(csv, cfg, "term_allreduce",  &tot.allreduce,  &b, &tot, sim_time);
        fclose(csv);
    }
    printf("Base station: all nodes terminated. %d alerts (%d with nearest nodes), %.0f neighbour messages. Logs in %s\n",
           b.alerts, b.replies_found, tot.nbr_msgs_sent, cfg->run_dir);
    fflush(stdout);

    free(treq);
    free(prep);
    free(preq);
    free(idx);
    free(b.last_report);
    free(b.last_full);
    free(b.last_silent);
    free(b.alerts_by_node);
    free(b.silent_by_node);
    free(b.reply_buf);
    free(b.reply_req);
    if (b.log != stderr) fclose(b.log);
}

/* ======================= main ======================= */
static void make_dir(const char *p) {
    if (mkdir(p, 0755) != 0 && errno != EEXIST) perror(p);
}

int main(int argc, char **argv) {
    int provided, world_rank, world_size;
    /* only the master thread of each process makes MPI calls */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.m              = arg_i(argc, argv, 1, 3);
    cfg.n              = arg_i(argc, argv, 2, 3);
    cfg.k              = arg_i(argc, argv, 3, 6);
    cfg.cycle          = arg_d(argc, argv, 4, 1.0);
    cfg.duration       = arg_d(argc, argv, 5, 30.0);
    cfg.threshold      = arg_i(argc, argv, 6, 2);
    cfg.busy_pct       = arg_i(argc, argv, 7, 60);
    cfg.variant        = arg_i(argc, argv, 8, VAR_ON_DEMAND);
    cfg.period         = arg_d(argc, argv, 9, 5.0);
    cfg.base_variant   = arg_i(argc, argv, 10, BV_IPROBE_RECV);
    cfg.poll_ms        = arg_d(argc, argv, 11, 1.0);
    cfg.nbr_timeout_ms = arg_i(argc, argv, 12, 500);
    cfg.dead_node      = arg_i(argc, argv, 13, -1);
    snprintf(cfg.out_dir, sizeof cfg.out_dir, "%s", argc > 14 ? argv[14] : "logs");

    int bad = 0;
    if (provided < MPI_THREAD_FUNNELED) bad = 1;
    else if (cfg.m < 1 || cfg.n < 1 || cfg.m * cfg.n + 1 != world_size) bad = 2;
    else if (cfg.k < 1 || cfg.k > MAX_PORTS) bad = 3;
    else if (cfg.variant < 0 || cfg.variant >= NUM_VARIANTS ||
             cfg.base_variant < 0 || cfg.base_variant >= NUM_BV) bad = 4;
    else if (cfg.cycle <= 0 || cfg.period < 0 || cfg.poll_ms <= 0 || cfg.nbr_timeout_ms <= 0) bad = 5;
    else if (cfg.dead_node >= cfg.m * cfg.n || (cfg.dead_node >= 0 && cfg.variant != VAR_ON_DEMAND)) bad = 6;
    if (bad) {
        if (world_rank == 0) {
            if (bad == 1) fprintf(stderr, "MPI_THREAD_FUNNELED not supported\n");
            if (bad == 2) fprintf(stderr, "need -np %d for a %dx%d grid (got %d)\n",
                                  cfg.m * cfg.n + 1, cfg.m, cfg.n, world_size);
            if (bad == 3) fprintf(stderr, "k must be 1..%d\n", MAX_PORTS);
            if (bad == 4) fprintf(stderr, "variant must be 0..%d and base_variant 0..%d\n",
                                  NUM_VARIANTS - 1, NUM_BV - 1);
            if (bad == 5) fprintf(stderr, "cycle, poll_ms and nbr_timeout_ms must be > 0, period >= 0\n");
            if (bad == 6) fprintf(stderr, "dead_node must be -1 or 0..%d, and needs variant 0 (on-demand)\n",
                                  cfg.m * cfg.n - 1);
        }
        MPI_Finalize();
        return 1;
    }

    /* one folder per run (timestamped) so CAAS runs never overwrite each other */
    if (world_rank == BASE_RANK) {
        char stamp[32];
        time_t now = time(NULL);
        struct tm tmv;
        localtime_r(&now, &tmv);
        strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", &tmv);
        snprintf(cfg.run_dir, sizeof cfg.run_dir, "%s/run_%dx%d_v%d_b%d_%s",
                 cfg.out_dir, cfg.m, cfg.n, cfg.variant, cfg.base_variant, stamp);
        make_dir(cfg.out_dir);
        make_dir(cfg.run_dir);
    }
    MPI_Bcast(cfg.run_dir, sizeof cfg.run_dir, MPI_CHAR, BASE_RANK, MPI_COMM_WORLD);

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
