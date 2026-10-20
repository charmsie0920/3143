/* FIT3143 Lab 3 - Distributed WSN of EV charging nodes (Open MPI + OpenMP)
 * Charmaine Chee Hing Yi 33520399 | Nazara Glaukofani Lopulisa 35135271
 *
 * compile: mpicc -fopenmp -Wall -o wsn wsn.c
 * run:     mpirun --oversubscribe -np <m*n+1> ./wsn m n [k cycle duration threshold busy_pct
 *                                                      variant nbr_timeout_ms dead_node out_dir]
 *          e.g. mpirun --oversubscribe -np 10 ./wsn 3 3 6 1.0 30 2 60 0
 *          low-availability test : busy_pct = 90
 *          dead-node test        : dead_node = 4  (node 4 stops answering its neighbours)
 *
 * variant 0 = ON-DEMAND (default): a node asks its neighbours ONLY when it is congested
 *             (TAG_NBR_REQ / TAG_NBR_RESP, non-blocking, with a timeout)
 * variant 1-3 = PERIODIC: every node exchanges with its neighbours EVERY cycle
 *             (1 = MPI_Neighbor_allgather, 2 = MPI_Ineighbor_allgather, 3 = Isend/Irecv)
 *             kept for comparison in Task 3.
 *
 * world rank 0 = base station; world ranks 1..m*n = charging nodes (cart rank = world rank - 1) */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
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
#define POLL_SECONDS    0.001      /* comm thread checks for messages every 1 ms */
#define BASE_TIMEOUT_S  5.0        /* give up waiting for the base station     */

#define NBR_NONE        -1         /* no neighbour in this direction (grid edge) */
#define NBR_NO_RESPONSE -2         /* neighbour did not answer before timeout    */

/* neighbour order = MPI_Neighbor_allgather order on a 2D cart comm */
enum { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3, NUM_DIRS = 4 };

/* one tag per message type */
enum {
    TAG_REPORT   = 1,   /* node -> base : congestion report              */
    TAG_REPLY    = 2,   /* base -> node : nearest available nodes        */
    TAG_TERM     = 3,   /* base -> node : terminate                      */
    TAG_DONE     = 4,   /* node -> base : node has shut down + its stats */
    TAG_NBR      = 5,   /* node <-> node: availability (periodic p2p)    */
    TAG_NBR_REQ  = 6,   /* node -> nbr  : "send me your availability"    */
    TAG_NBR_RESP = 7    /* nbr -> node  : availability                   */
};

enum { VAR_ON_DEMAND = 0,                /* request/response only when congested */
       VAR_PERIODIC_NEIGHBOR_ALLGATHER,  /* blocking neighbourhood collective    */
       VAR_PERIODIC_INEIGHBOR_ALLGATHER, /* non-blocking neighbourhood coll.     */
       VAR_PERIODIC_P2P,                 /* non-blocking point-to-point          */
       NUM_VARIANTS };
static const char *variant_name[NUM_VARIANTS] = { "on-demand", "periodic-neighbor_allgather",
                                                  "periodic-ineighbor_allgather", "periodic-p2p" };

typedef struct {
    int m, n, k;             /* grid rows, cols, ports per node              */
    int threshold;           /* congested when free ports < threshold        */
    int busy_pct;            /* chance (%) a port is in use each update      */
    int variant;
    int nbr_timeout_ms;      /* on-demand: wait this long for neighbours     */
    int dead_node;           /* cart rank that stops responding, -1 = none   */
    double cycle;            /* seconds per simulation cycle                 */
    double duration;         /* seconds the base station runs the simulation */
    char out_dir[200];
    char run_dir[320];
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
    double nbr_comm_time;                           /* neighbour round (s), measured on this node */
    double last_base_rtt;                           /* previous alert round trip (s), -1 = none   */
} Report;

typedef struct {
    int count;                       /* 0 = no nodes available nearby */
    int distance;                    /* Manhattan distance of results */
    int rank[MAX_CAND];              /* cart ranks                    */
    int coords[MAX_CAND][2];
    double base_proc_time;           /* time base spent on this report (s) */
} Reply;

/* statistics each node sends to the base station with TAG_DONE (as MPI_DOUBLEs) */
enum { ST_CYCLES, ST_CONGESTED, ST_LOCAL_REDIRECTS, ST_REPORTS, ST_REPLIES,
       ST_NBR_MSGS_SENT, ST_NBR_MSGS_RECV, ST_NBR_BYTES,
       ST_NBR_ROUNDS, ST_NBR_ROUND_SUM, ST_NBR_ROUND_MAX,
       ST_NBR_RTT_COUNT, ST_NBR_RTT_SUM, ST_NBR_RTT_MAX,
       ST_BASE_RTT_COUNT, ST_BASE_RTT_SUM, ST_BASE_RTT_MAX,
       ST_NBR_TIMEOUTS, ST_BASE_TIMEOUTS, NSTAT };

/* ======================= MPI derived datatypes ======================= */
/* Report: block of ints followed by block of doubles */
MPI_Datatype build_report_type(void) {
    int n_ints = (int)((offsetof(Report, nbr_comm_time) - offsetof(Report, cart_rank)) / sizeof(int));
    int blocklens[2] = { n_ints, 2 };
    MPI_Aint disps[2] = { offsetof(Report, cart_rank), offsetof(Report, nbr_comm_time) };
    MPI_Datatype types[2] = { MPI_INT, MPI_DOUBLE }, tmp, t;
    MPI_Type_create_struct(2, blocklens, disps, types, &tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(Report), &t);   /* honour trailing padding */
    MPI_Type_free(&tmp);
    MPI_Type_commit(&t);
    return t;
}

MPI_Datatype build_reply_type(void) {
    int n_ints = (int)((offsetof(Reply, base_proc_time) - offsetof(Reply, count)) / sizeof(int));
    int blocklens[2] = { n_ints, 1 };
    MPI_Aint disps[2] = { offsetof(Reply, count), offsetof(Reply, base_proc_time) };
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

static void stat_time(double *st, int count_i, int sum_i, int max_i, double v) {
    st[count_i] += 1; st[sum_i] += v;
    if (v > st[max_i]) st[max_i] = v;
}

/* ======================= TASK 1: EV charging node ======================= */
/* ---------- state shared by all threads of this process ---------- */
static int        port_state[MAX_PORTS];     /* 1 = in use, 0 = free      */
static int        history[HISTORY_LEN];      /* availability per cycle    */
static int        history_count = 0;
static omp_lock_t state_lock;                /* guards port_state/history */
static int        g_terminate = 0;           /* set by comm thread        */

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
    double st[NSTAT];
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

static const char *dir_name[NUM_DIRS] = { "up", "down", "left", "right" };

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
    MPI_Isend(r, 1, nd->report_t, BASE_RANK, TAG_REPORT, MPI_COMM_WORLD, &nd->report_req);
    nd->waiting_base = 1;
    nd->st[ST_REPORTS]++;
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
            nd->st[ST_NBR_MSGS_RECV]++; nd->st[ST_NBR_BYTES] += sizeof msg;
            if (answer && d >= 0) {
                MPI_Wait(&nd->resp_req[d], MPI_STATUS_IGNORE);   /* previous answer delivered */
                nd->resp_buf[d][0] = count_free(nd->cfg->k);
                nd->resp_buf[d][1] = msg[1];                     /* echo the round id */
                MPI_Isend(nd->resp_buf[d], 2, MPI_INT, s.MPI_SOURCE, TAG_NBR_RESP, nd->cart, &nd->resp_req[d]);
                nd->st[ST_NBR_MSGS_SENT]++; nd->st[ST_NBR_BYTES] += sizeof msg;
            }
        } else if (s.MPI_TAG == TAG_NBR_RESP) {
            MPI_Recv(msg, 2, MPI_INT, s.MPI_SOURCE, TAG_NBR_RESP, nd->cart, MPI_STATUS_IGNORE);
            nd->st[ST_NBR_MSGS_RECV]++; nd->st[ST_NBR_BYTES] += sizeof msg;
            /* ignore late answers that belong to an older round */
            if (rd && rd->active && d >= 0 && msg[1] == rd->id && !rd->responded[d]) {
                rd->responded[d] = 1;
                rd->avail[d] = msg[0];
                rd->msgs++; rd->bytes += sizeof msg;
                stat_time(nd->st, ST_NBR_RTT_COUNT, ST_NBR_RTT_SUM, ST_NBR_RTT_MAX, MPI_Wtime() - rd->t_start);
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
        nd->st[ST_NBR_MSGS_SENT]++; nd->st[ST_NBR_BYTES] += sizeof nd->req_buf;
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
    stat_time(nd->st, ST_NBR_ROUNDS, ST_NBR_ROUND_SUM, ST_NBR_ROUND_MAX, xchg);

    for (int d = 0; d < NUM_DIRS; d++)
        if (rd->avail[d] == NBR_NO_RESPONSE) {
            nd->st[ST_NBR_TIMEOUTS]++;
            fprintf(nd->log, "  WARNING neighbour %s node %d (%d,%d) did not respond within %d ms - possibly dead\n",
                    dir_name[d], nd->nbr[d], nd->nbr_coords[d][0], nd->nbr_coords[d][1], nd->cfg->nbr_timeout_ms);
        }

    char action[128];
    int best = best_neighbour(nd, rd->avail);
    if (best >= 0) {
        /* a neighbour has room: redirect locally, no base station traffic */
        nd->st[ST_LOCAL_REDIRECTS]++;
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
    nd->st[ST_NBR_MSGS_SENT] += nd->num_nbrs;
    nd->st[ST_NBR_MSGS_RECV] += nd->num_nbrs;
    nd->st[ST_NBR_BYTES]     += 2.0 * nd->num_nbrs * sizeof(int);
}

/* ---------- message from the base station ---------- */
static void handle_base_message(Node *nd, MPI_Status *s, int *term) {
    if (s->MPI_TAG == TAG_REPLY) {
        Reply reply;
        MPI_Recv(&reply, 1, nd->reply_t, BASE_RANK, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double rtt = MPI_Wtime() - nd->t_alert;              /* both times taken on this node */
        nd->last_base_rtt = rtt;
        nd->waiting_base = 0;
        nd->st[ST_REPLIES]++;
        stat_time(nd->st, ST_BASE_RTT_COUNT, ST_BASE_RTT_SUM, ST_BASE_RTT_MAX, rtt);
        if (reply.count == 0) {
            fprintf(nd->log, "  [base reply] no nodes available nearby (round trip %.3f ms)\n", rtt * 1e3);
        } else {
            fprintf(nd->log, "  [base reply] forward EVs to %d node(s) at distance %d:", reply.count, reply.distance);
            for (int i = 0; i < reply.count; i++)
                fprintf(nd->log, " node%d(%d,%d)", reply.rank[i], reply.coords[i][0], reply.coords[i][1]);
            fprintf(nd->log, " (round trip %.3f ms)\n", rtt * 1e3);
        }
    } else if (s->MPI_TAG == TAG_TERM) {
        int term_msg;
        MPI_Recv(&term_msg, 1, MPI_INT, BASE_RANK, TAG_TERM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    int on_demand = (cfg->variant == VAR_ON_DEMAND);

    fprintf(nd->log, "node cart=%d world=%d coords=(%d,%d) k=%d threshold=%d variant=%d (%s)%s\n",
            nd->cart_rank, nd->cart_rank + 1, nd->coords[0], nd->coords[1], cfg->k, cfg->threshold,
            cfg->variant, variant_name[cfg->variant], nd->dead ? "  ** SIMULATED DEAD NODE **" : "");
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
            nd->st[ST_BASE_TIMEOUTS]++;
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
                MPI_Allreduce(&term, &stop, 1, MPI_INT, MPI_MAX, nd->cart);
                if (stop) break;
            }

            cycle++;
            nd->st[ST_CYCLES]++;
            int avail = snapshot_availability(cfg->k, states);
            time_t t = time(NULL);
            struct tm tmv;
            localtime_r(&t, &tmv);
            int congested = avail < cfg->threshold;
            if (congested) nd->st[ST_CONGESTED]++;

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
                stat_time(nd->st, ST_NBR_ROUNDS, ST_NBR_ROUND_SUM, ST_NBR_ROUND_MAX, xchg);

                char action[128] = "ok";
                if (congested) {
                    int best = best_neighbour(nd, nbr_avail);
                    if (best >= 0) {
                        nd->st[ST_LOCAL_REDIRECTS]++;
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
        sleep_seconds(POLL_SECONDS);
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
            if (!done) sleep_seconds(POLL_SECONDS);
        }
        for (double end = MPI_Wtime() + 0.02; MPI_Wtime() < end; sleep_seconds(POLL_SECONDS))
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
    MPI_Send(nd->st, NSTAT, MPI_DOUBLE, BASE_RANK, TAG_DONE, MPI_COMM_WORLD);

    const double *st = nd->st;
    fprintf(nd->log, "\nTERMINATED by base station\n");
    fprintf(nd->log, "summary: cycles=%.0f congested=%.0f local_redirects=%.0f reports_to_base=%.0f replies=%.0f\n",
            st[ST_CYCLES], st[ST_CONGESTED], st[ST_LOCAL_REDIRECTS], st[ST_REPORTS], st[ST_REPLIES]);
    fprintf(nd->log, "         neighbour msgs sent=%.0f recv=%.0f bytes=%.0f rounds=%.0f avg_round=%.3f ms timeouts=%.0f\n",
            st[ST_NBR_MSGS_SENT], st[ST_NBR_MSGS_RECV], st[ST_NBR_BYTES], st[ST_NBR_ROUNDS],
            st[ST_NBR_ROUNDS] ? st[ST_NBR_ROUND_SUM] / st[ST_NBR_ROUNDS] * 1e3 : 0.0, st[ST_NBR_TIMEOUTS]);
    fprintf(nd->log, "         base round trip avg=%.3f ms max=%.3f ms\n",
            st[ST_BASE_RTT_COUNT] ? st[ST_BASE_RTT_SUM] / st[ST_BASE_RTT_COUNT] * 1e3 : 0.0, st[ST_BASE_RTT_MAX] * 1e3);
}

void node_run(const Config *cfg, MPI_Comm cart, MPI_Datatype report_t, MPI_Datatype reply_t) {
    static Node nd;                                    /* large (holds a Report): keep off the stack */
    memset(&nd, 0, sizeof nd);
    nd.cfg = cfg; nd.cart = cart; nd.report_t = report_t; nd.reply_t = reply_t;
    MPI_Comm_rank(cart, &nd.cart_rank);

    char fname[400];
    snprintf(fname, sizeof fname, "%s/node_%d.log", cfg->run_dir, nd.cart_rank);
    nd.log = fopen(fname, "w");
    if (!nd.log) nd.log = stderr;

    omp_init_lock(&state_lock);
    memset(port_state, 0, sizeof port_state);
    g_terminate = 0;

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

/* ======================= BASE STATION (placeholder for Task 2) =======================
 * Iprobe -> Recv reports, logs them, replies "no nodes available", terminates all nodes
 * after duration, collects statistics. TODO Task 2: compute the nearest available nodes. */
static void log_report(FILE *log, const Report *r, double t) {
    fprintf(log, "------------------------------------------------------------\n");
    fprintf(log, "[%8.3f s] ALERT from node %d (%d,%d), detected %04d-%02d-%02d %02d:%02d:%02d, cycle %d\n",
            t, r->cart_rank, r->coords[0], r->coords[1], r->year, r->month, r->day,
            r->hour, r->minute, r->second, r->cycle);
    fprintf(log, "  availability : %d/%d free, ports [", r->availability, r->k);
    for (int i = 0; i < r->k && i < MAX_PORTS; i++) fputc(r->port_state[i] ? '1' : '0', log);
    fprintf(log, "]\n  history      :");
    for (int i = 0; i < HISTORY_LEN; i++) if (r->history[i] >= 0) fprintf(log, " %d", r->history[i]);
    fprintf(log, "\n  neighbours   : %d\n", r->num_neighbours);
    for (int d = 0; d < NUM_DIRS; d++) {
        if (r->nbr_rank[d] < 0) continue;
        if (r->nbr_avail[d] == NBR_NO_RESPONSE)
            fprintf(log, "    %-5s node %d (%d,%d): NO RESPONSE (possibly dead)\n",
                    dir_name[d], r->nbr_rank[d], r->nbr_coords[d][0], r->nbr_coords[d][1]);
        else
            fprintf(log, "    %-5s node %d (%d,%d): %d free\n",
                    dir_name[d], r->nbr_rank[d], r->nbr_coords[d][0], r->nbr_coords[d][1], r->nbr_avail[d]);
    }
    fprintf(log, "  neighbour comm: %d messages, %d bytes, %.3f ms\n",
            r->msgs_with_neighbours, r->bytes_with_neighbours, r->nbr_comm_time * 1e3);
    if (r->last_base_rtt >= 0)
        fprintf(log, "  previous alert round trip (measured by node): %.3f ms\n", r->last_base_rtt * 1e3);
}

void base_run(const Config *cfg, MPI_Datatype report_t, MPI_Datatype reply_t) {
    int num_nodes = cfg->m * cfg->n;
    char fname[400];
    snprintf(fname, sizeof fname, "%s/base_station.log", cfg->run_dir);
    FILE *log = fopen(fname, "w");
    if (!log) log = stderr;
    fprintf(log, "BASE STUB grid=%dx%d k=%d threshold=%d busy=%d%% cycle=%.2fs duration=%.1fs variant=%d (%s)\n",
            cfg->m, cfg->n, cfg->k, cfg->threshold, cfg->busy_pct, cfg->cycle, cfg->duration,
            cfg->variant, variant_name[cfg->variant]);
    printf("Base station: %dx%d grid, %d nodes, %s, running %.0f s\n",
           cfg->m, cfg->n, num_nodes, variant_name[cfg->variant], cfg->duration);
    fflush(stdout);

    Report rep;
    Reply *reply = calloc(num_nodes, sizeof(Reply));          /* one buffer per node */
    MPI_Request *reply_req = malloc(num_nodes * sizeof(MPI_Request));
    double (*node_st)[NSTAT] = calloc(num_nodes, sizeof *node_st);
    for (int i = 0; i < num_nodes; i++) reply_req[i] = MPI_REQUEST_NULL;
    int alerts = 0, late = 0;
    double proc_sum = 0, proc_max = 0;
    double start = MPI_Wtime();

    /* listen: Iprobe, and only call Recv when a message is there */
    while (MPI_Wtime() - start < cfg->duration) {
        int flag = 0;
        MPI_Status st;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
        if (!flag) { sleep_seconds(POLL_SECONDS); continue; }

        if (st.MPI_TAG == TAG_REPORT) {
            MPI_Recv(&rep, 1, report_t, st.MPI_SOURCE, TAG_REPORT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double t0 = MPI_Wtime();
            int node = st.MPI_SOURCE - 1;
            alerts++;
            log_report(log, &rep, t0 - start);

            MPI_Wait(&reply_req[node], MPI_STATUS_IGNORE);     /* previous reply to this node delivered */
            memset(&reply[node], 0, sizeof(Reply));             /* TODO Task 2: nearest available nodes */
            reply[node].base_proc_time = MPI_Wtime() - t0;
            MPI_Isend(&reply[node], 1, reply_t, st.MPI_SOURCE, TAG_REPLY, MPI_COMM_WORLD, &reply_req[node]);
            proc_sum += reply[node].base_proc_time;
            if (reply[node].base_proc_time > proc_max) proc_max = reply[node].base_proc_time;
            fprintf(log, "  reply        : no nodes available nearby (stub)\n");
        } else {
            int cnt; char junk[1024];
            MPI_Get_count(&st, MPI_BYTE, &cnt);
            MPI_Recv(junk, cnt > 1024 ? 1024 : cnt, MPI_BYTE, st.MPI_SOURCE, st.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            fprintf(log, "WARNING unexpected tag %d from rank %d\n", st.MPI_TAG, st.MPI_SOURCE);
        }
    }
    double sim_time = MPI_Wtime() - start;

    /* termination message to every node (point-to-point: the base station is not in the cart comm) */
    printf("Base station: time up, sending TERMINATE to %d nodes\n", num_nodes);
    fflush(stdout);
    MPI_Request *treq = malloc(num_nodes * sizeof(MPI_Request));
    int term = 1;
    for (int i = 0; i < num_nodes; i++)
        MPI_Isend(&term, 1, MPI_INT, i + 1, TAG_TERM, MPI_COMM_WORLD, &treq[i]);
    MPI_Waitall(num_nodes, treq, MPI_STATUSES_IGNORE);

    /* drain in-flight reports until every node confirms shutdown */
    int done = 0;
    while (done < num_nodes) {
        int flag = 0;
        MPI_Status st;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
        if (!flag) { sleep_seconds(POLL_SECONDS); continue; }
        if (st.MPI_TAG == TAG_DONE) {
            MPI_Recv(node_st[st.MPI_SOURCE - 1], NSTAT, MPI_DOUBLE, st.MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            done++;
        } else {
            MPI_Recv(&rep, 1, report_t, st.MPI_SOURCE, TAG_REPORT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            fprintf(log, "late report from node %d after termination - discarded\n", rep.cart_rank);
            late++;
        }
    }
    MPI_Waitall(num_nodes, reply_req, MPI_STATUSES_IGNORE);

    /* ---------- summary (Task 3 data) ---------- */
    double tot[NSTAT] = { 0 };
    for (int i = 0; i < num_nodes; i++)
        for (int j = 0; j < NSTAT; j++) {
            if (j == ST_NBR_ROUND_MAX || j == ST_NBR_RTT_MAX || j == ST_BASE_RTT_MAX) {
                if (node_st[i][j] > tot[j]) tot[j] = node_st[i][j];
            } else tot[j] += node_st[i][j];
        }
    double round_mean = tot[ST_NBR_ROUNDS] ? tot[ST_NBR_ROUND_SUM] / tot[ST_NBR_ROUNDS] * 1e3 : 0;
    double rtt_mean   = tot[ST_NBR_RTT_COUNT] ? tot[ST_NBR_RTT_SUM] / tot[ST_NBR_RTT_COUNT] * 1e3 : 0;
    double base_mean  = tot[ST_BASE_RTT_COUNT] ? tot[ST_BASE_RTT_SUM] / tot[ST_BASE_RTT_COUNT] * 1e3 : 0;
    double proc_mean  = alerts ? proc_sum / alerts * 1e3 : 0;
    long base_msgs = alerts + late + alerts /* replies */ + 2L * num_nodes /* TERM + DONE */;

    fprintf(log, "============================================================\n");
    fprintf(log, "TERMINATED: sim_time=%.3fs alerts=%d late_reports_drained=%d\n", sim_time, alerts, late);
    fprintf(log, "cycles=%.0f congested=%.0f local_redirects=%.0f reports=%.0f replies=%.0f\n",
            tot[ST_CYCLES], tot[ST_CONGESTED], tot[ST_LOCAL_REDIRECTS], tot[ST_REPORTS], tot[ST_REPLIES]);
    fprintf(log, "neighbour messages: sent=%.0f recv=%.0f bytes=%.0f rounds=%.0f timeouts=%.0f\n",
            tot[ST_NBR_MSGS_SENT], tot[ST_NBR_MSGS_RECV], tot[ST_NBR_BYTES], tot[ST_NBR_ROUNDS], tot[ST_NBR_TIMEOUTS]);
    fprintf(log, "neighbour round time   : mean %.3f ms  max %.3f ms\n", round_mean, tot[ST_NBR_ROUND_MAX] * 1e3);
    fprintf(log, "neighbour round trip   : mean %.3f ms  max %.3f ms  (n=%.0f, on-demand only)\n",
            rtt_mean, tot[ST_NBR_RTT_MAX] * 1e3, tot[ST_NBR_RTT_COUNT]);
    fprintf(log, "alert->reply round trip: mean %.3f ms  max %.3f ms  (n=%.0f)\n",
            base_mean, tot[ST_BASE_RTT_MAX] * 1e3, tot[ST_BASE_RTT_COUNT]);
    fprintf(log, "base processing/alert  : mean %.4f ms  max %.4f ms\n", proc_mean, proc_max * 1e3);
    fprintf(log, "messages with base     : %ld\n", base_msgs);

    char csv_path[300];
    snprintf(csv_path, sizeof csv_path, "%s/summary.csv", cfg->out_dir);
    int is_new = access(csv_path, F_OK) != 0;
    FILE *csv = fopen(csv_path, "a");
    if (csv) {
        if (is_new)
            fprintf(csv, "m,n,nodes,k,threshold,busy_pct,variant,cycle_s,duration_s,dead_node,cycles,congested,"
                         "local_redirects,reports,late_reports,replies,nbr_msgs_sent,nbr_bytes,nbr_rounds,"
                         "nbr_round_mean_ms,nbr_round_max_ms,nbr_rtt_mean_ms,nbr_rtt_max_ms,"
                         "base_rtt_count,base_rtt_mean_ms,base_rtt_max_ms,base_proc_mean_ms,nbr_timeouts,base_msgs\n");
        fprintf(csv, "%d,%d,%d,%d,%d,%d,%s,%.2f,%.1f,%d,%.0f,%.0f,%.0f,%.0f,%d,%.0f,%.0f,%.0f,%.0f,"
                     "%.4f,%.4f,%.4f,%.4f,%.0f,%.4f,%.4f,%.4f,%.0f,%ld\n",
                cfg->m, cfg->n, num_nodes, cfg->k, cfg->threshold, cfg->busy_pct, variant_name[cfg->variant],
                cfg->cycle, cfg->duration, cfg->dead_node, tot[ST_CYCLES], tot[ST_CONGESTED],
                tot[ST_LOCAL_REDIRECTS], tot[ST_REPORTS], late, tot[ST_REPLIES], tot[ST_NBR_MSGS_SENT],
                tot[ST_NBR_BYTES], tot[ST_NBR_ROUNDS], round_mean, tot[ST_NBR_ROUND_MAX] * 1e3,
                rtt_mean, tot[ST_NBR_RTT_MAX] * 1e3, tot[ST_BASE_RTT_COUNT], base_mean,
                tot[ST_BASE_RTT_MAX] * 1e3, proc_mean, tot[ST_NBR_TIMEOUTS], base_msgs);
        fclose(csv);
    }
    printf("Base station: all nodes terminated. %d alerts, %.0f neighbour messages. Logs in %s\n",
           alerts, tot[ST_NBR_MSGS_SENT], cfg->run_dir);
    fflush(stdout);

    if (log != stderr) fclose(log);
    free(reply); free(reply_req); free(node_st); free(treq);
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
    cfg.nbr_timeout_ms = arg_i(argc, argv, 9, 500);
    cfg.dead_node      = arg_i(argc, argv, 10, -1);
    snprintf(cfg.out_dir, sizeof cfg.out_dir, "%s", argc > 11 ? argv[11] : "logs");

    int bad = 0;
    if (provided < MPI_THREAD_FUNNELED) bad = 1;
    else if (cfg.m < 1 || cfg.n < 1 || cfg.m * cfg.n + 1 != world_size) bad = 2;
    else if (cfg.k < 1 || cfg.k > MAX_PORTS) bad = 3;
    else if (cfg.variant < 0 || cfg.variant >= NUM_VARIANTS) bad = 4;
    else if (cfg.dead_node >= cfg.m * cfg.n || (cfg.dead_node >= 0 && cfg.variant != VAR_ON_DEMAND)) bad = 5;
    else if (cfg.cycle <= 0 || cfg.nbr_timeout_ms <= 0) bad = 6;
    if (bad) {
        if (world_rank == 0) {
            if (bad == 1) fprintf(stderr, "MPI_THREAD_FUNNELED not supported\n");
            if (bad == 2) fprintf(stderr, "need -np %d for a %dx%d grid (got %d)\n",
                                  cfg.m * cfg.n + 1, cfg.m, cfg.n, world_size);
            if (bad == 3) fprintf(stderr, "k must be 1..%d\n", MAX_PORTS);
            if (bad == 4) fprintf(stderr, "variant must be 0..%d\n", NUM_VARIANTS - 1);
            if (bad == 5) fprintf(stderr, "dead_node must be -1 or 0..%d, and needs variant 0 (on-demand)\n", cfg.m * cfg.n - 1);
            if (bad == 6) fprintf(stderr, "cycle and nbr_timeout_ms must be > 0\n");
        }
        MPI_Finalize();
        return 1;
    }

    /* one folder per run so CAAS runs do not overwrite each other */
    snprintf(cfg.run_dir, sizeof cfg.run_dir, "%s/run_%dx%d_v%d", cfg.out_dir, cfg.m, cfg.n, cfg.variant);
    if (world_rank == BASE_RANK) { make_dir(cfg.out_dir); make_dir(cfg.run_dir); }
    MPI_Barrier(MPI_COMM_WORLD);                     /* folder exists before anyone opens a log */

    /* base station is NOT part of the grid: split it out (star vs mesh) */
    MPI_Comm nodes_comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank == BASE_RANK ? MPI_UNDEFINED : 0,
                   world_rank, &nodes_comm);

    MPI_Datatype report_t = build_report_type();
    MPI_Datatype reply_t  = build_reply_type();

    if (world_rank == BASE_RANK) {
        base_run(&cfg, report_t, reply_t);           /* nodes_comm == MPI_COMM_NULL */
    } else {
        int dims[2] = { cfg.m, cfg.n }, periods[2] = { 0, 0 };   /* mesh, not torus */
        MPI_Comm cart_comm;
        MPI_Cart_create(nodes_comm, 2, dims, periods, 0 /* no reorder */, &cart_comm);
        node_run(&cfg, cart_comm, report_t, reply_t);
        MPI_Comm_free(&cart_comm);                   /* collective over nodes only */
        MPI_Comm_free(&nodes_comm);
    }

    MPI_Type_free(&report_t);
    MPI_Type_free(&reply_t);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
