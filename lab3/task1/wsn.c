/* FIT3143 Lab 3 - Distributed WSN of EV charging nodes (Open MPI + OpenMP)
 * Charmaine Chee Hing Yi 33520399 | Nazara Glaukofani Lopulisa 35135271
 *
 * compile: mpicc -fopenmp -Wall -o wsn wsn.c
 * run:     mpirun --oversubscribe -np <m*n+1> ./wsn m n [k cycle duration threshold busy_pct variant]
 *          e.g. mpirun --oversubscribe -np 10 ./wsn 3 3 6 1.0 30 2 60 0
 *          low-availability test: busy_pct = 90
 *
 * world rank 0 = base station; world ranks 1..m*n = charging nodes (cart rank = world rank - 1) */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

/* ======================= shared definitions ======================= */

#define BASE_RANK   0          /* world rank of the base station        */
#define MAX_PORTS   64
#define MAX_CAND    64         /* max nearest nodes in a redirect reply */
#define HISTORY_LEN 16         /* circular buffer of availability       */

/* neighbour order = MPI_Neighbor_allgather order on a 2D cart comm */
enum { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3, NUM_DIRS = 4 };

enum {
    TAG_REPORT = 1,   /* node -> base : congestion report              */
    TAG_REPLY  = 2,   /* base -> node : nearest available nodes        */
    TAG_TERM   = 3,   /* base -> node : terminate                      */
    TAG_DONE   = 4,   /* node -> base : node has shut down its traffic */
    TAG_NBR    = 5    /* node <-> node: availability (p2p variant)     */
};

/* neighbour-exchange implementations compared in Task 3 */
enum { VAR_NEIGHBOR_ALLGATHER = 0,   /* blocking neighbourhood collective    */
       VAR_INEIGHBOR_ALLGATHER = 1,  /* non-blocking neighbourhood coll.    */
       VAR_P2P_ISEND_IRECV = 2 };    /* non-blocking point-to-point         */

typedef struct {
    int m, n, k;             /* grid rows, cols, ports per node              */
    int threshold;           /* congested when free ports < threshold        */
    int busy_pct;            /* chance (%) a port is in use each update      */
    int variant;
    double cycle;            /* seconds per simulation cycle                 */
    double duration;         /* seconds the base station runs the simulation */
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
    double send_wtime;                              /* MPI_Wtime at send      */
    double nbr_comm_time;                           /* neighbour exchange (s) */
} Report;

typedef struct {
    int count;                       /* 0 = no nodes available nearby */
    int distance;                    /* Manhattan distance of results */
    int rank[MAX_CAND];              /* cart ranks                    */
    int coords[MAX_CAND][2];
    double send_wtime;
} Reply;

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
    int n_ints = (int)((offsetof(Reply, send_wtime) - offsetof(Reply, count)) / sizeof(int));
    int blocklens[2] = { n_ints, 1 };
    MPI_Aint disps[2] = { offsetof(Reply, count), offsetof(Reply, send_wtime) };
    MPI_Datatype types[2] = { MPI_INT, MPI_DOUBLE }, tmp, t;
    MPI_Type_create_struct(2, blocklens, disps, types, &tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(Reply), &t);
    MPI_Type_free(&tmp);
    MPI_Type_commit(&t);
    return t;
}

static int arg_i(int argc, char **argv, int i, int def) { return argc > i ? atoi(argv[i]) : def; }
static double arg_d(int argc, char **argv, int i, double def) { return argc > i ? atof(argv[i]) : def; }

/* ======================= TASK 1: EV charging node ======================= */
/* ---------- state shared by all threads of this process ---------- */
static int        port_state[MAX_PORTS];     /* 1 = in use, 0 = free      */
static int        history[HISTORY_LEN];      /* availability per cycle    */
static int        history_count = 0;
static omp_lock_t state_lock;                /* guards port_state/history */
static int        g_terminate = 0;           /* set by comm thread        */

static void sleep_seconds(double s) {
    if (s <= 0) return;
    struct timespec ts;
    ts.tv_sec = (time_t)s;
    ts.tv_nsec = (long)((s - (double)ts.tv_sec) * 1e9);
    nanosleep(&ts, NULL);
}

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

static const char *dir_name[NUM_DIRS] = { "up", "down", "left", "right" };

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

    fprintf(log, "node cart=%d world=%d coords=(%d,%d) k=%d threshold=%d variant=%d\n",
            cart_rank, cart_rank + 1, coords[0], coords[1], cfg->k, cfg->threshold, cfg->variant);
    for (int d = 0; d < NUM_DIRS; d++) {
        if (nbr[d] == MPI_PROC_NULL) fprintf(log, "  neighbour %-5s : none\n", dir_name[d]);
        else fprintf(log, "  neighbour %-5s : node %d (%d,%d)\n", dir_name[d], nbr[d],
                     nbr_coords[d][0], nbr_coords[d][1]);
    }
    fprintf(log, "YYYY MM DD HH MM SS | avail | ports(1=used) | nbr avail u/d/l/r | xchg(ms) | action\n");

    /* termination message can arrive at any time: keep a receive posted */
    int term_msg = 0, term_flag = 0;
    MPI_Request term_req, reply_req = MPI_REQUEST_NULL, report_req = MPI_REQUEST_NULL;
    MPI_Irecv(&term_msg, 1, MPI_INT, BASE_RANK, TAG_TERM, MPI_COMM_WORLD, &term_req);

    Report report;
    Reply  reply;
    int    states[MAX_PORTS], nbr_avail[NUM_DIRS];
    int    reports_sent = 0, redirects_local = 0, replies_received = 0;
    double total_xchg = 0;
    int    xchg_count = 0;

    for (int cycle = 0; ; cycle++) {
        double cycle_start = MPI_Wtime();

        /* 1. termination: all nodes must leave the loop in the SAME cycle,
         *    otherwise a node would block in a neighbour exchange forever */
        if (!term_flag) MPI_Test(&term_req, &term_flag, MPI_STATUS_IGNORE);
        int stop = 0;
        MPI_Allreduce(&term_flag, &stop, 1, MPI_INT, MPI_MAX, cart);
        if (stop) break;

        /* 2. read shared array */
        int avail = snapshot_availability(cfg->k, states);
        time_t now = time(NULL);
        struct tm tmv;
        localtime_r(&now, &tmv);

        /* 3. neighbour exchange (timed) */
        double t0 = MPI_Wtime();
        int num_nbrs = exchange_with_neighbours(cfg, cart, nbr, avail, nbr_avail);
        double xchg = MPI_Wtime() - t0;
        total_xchg += xchg;
        xchg_count++;

        /* 4. did the base station answer an earlier report? */
        if (reply_req != MPI_REQUEST_NULL) {
            int done = 0;
            MPI_Test(&reply_req, &done, MPI_STATUS_IGNORE);
            if (done) {
                replies_received++;
                double delay = MPI_Wtime() - reply.send_wtime;
                if (reply.count == 0) {
                    fprintf(log, "  [base reply] no nodes available nearby (delay %.3f ms)\n", delay * 1e3);
                } else {
                    fprintf(log, "  [base reply] forward EVs to %d node(s) at distance %d:",
                            reply.count, reply.distance);
                    for (int i = 0; i < reply.count; i++)
                        fprintf(log, " node%d(%d,%d)", reply.rank[i], reply.coords[i][0], reply.coords[i][1]);
                    fprintf(log, " (delay %.3f ms)\n", delay * 1e3);
                }
            }
        }

        /* 5. decision */
        char action[96] = "ok";
        if (avail < cfg->threshold) {
            int best = -1;
            for (int d = 0; d < NUM_DIRS; d++)
                if (nbr_avail[d] >= cfg->threshold && (best < 0 || nbr_avail[d] > nbr_avail[best]))
                    best = d;

            if (best >= 0) {
                /* a neighbour has room: redirect locally, no base station traffic */
                redirects_local++;
                snprintf(action, sizeof action, "CONGESTED -> redirect EVs to node %d (%d,%d) avail=%d",
                         nbr[best], nbr_coords[best][0], nbr_coords[best][1], nbr_avail[best]);
            } else if (reply_req != MPI_REQUEST_NULL) {
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
                MPI_Irecv(&reply, 1, reply_t, BASE_RANK, TAG_REPLY, MPI_COMM_WORLD, &reply_req);
                report.send_wtime = MPI_Wtime();
                MPI_Isend(&report, 1, report_t, BASE_RANK, TAG_REPORT, MPI_COMM_WORLD, &report_req);
                reports_sent++;
                snprintf(action, sizeof action, "CONGESTED, all neighbours full -> ALERT base station");
            }
        }

        fprintf(log, "%04d %02d %02d %02d %02d %02d | %5d | ",
                tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec, avail);
        for (int i = 0; i < cfg->k; i++) fputc(states[i] ? '1' : '0', log);
        fprintf(log, " | %2d %2d %2d %2d | %8.3f | %s\n",
                nbr_avail[UP], nbr_avail[DOWN], nbr_avail[LEFT], nbr_avail[RIGHT], xchg * 1e3, action);

        /* 6. wait for the rest of the cycle */
        sleep_seconds(cfg->cycle - (MPI_Wtime() - cycle_start));
    }

    /* ---------- graceful shutdown of communication ---------- */
    #pragma omp atomic write
    g_terminate = 1;                                   /* port threads exit their loops */

    if (!term_flag) MPI_Wait(&term_req, MPI_STATUS_IGNORE);   /* base sent it to everyone */
    MPI_Wait(&report_req, MPI_STATUS_IGNORE);
    if (reply_req != MPI_REQUEST_NULL) {               /* reply that will never come */
        MPI_Cancel(&reply_req);
        MPI_Wait(&reply_req, MPI_STATUS_IGNORE);
    }
    /* tell the base station we are done; it drains reports until all DONEs arrive */
    MPI_Send(&reports_sent, 1, MPI_INT, BASE_RANK, TAG_DONE, MPI_COMM_WORLD);

    fprintf(log, "\nTERMINATED by base station\n");
    fprintf(log, "summary: cycles=%d reports_to_base=%d replies=%d local_redirects=%d avg_xchg=%.3f ms\n",
            xchg_count, reports_sent, replies_received, redirects_local,
            xchg_count ? total_xchg / xchg_count * 1e3 : 0.0);
}

void node_run(const Config *cfg, MPI_Comm cart, MPI_Datatype report_t, MPI_Datatype reply_t) {
    int cart_rank;
    MPI_Comm_rank(cart, &cart_rank);

    char fname[64];
    snprintf(fname, sizeof fname, "node_%d.log", cart_rank);
    FILE *log = fopen(fname, "w");
    if (!log) log = stderr;

    omp_init_lock(&state_lock);
    memset(port_state, 0, sizeof port_state);
    g_terminate = 0;

    #pragma omp parallel num_threads(cfg->k + 1)
    {
        int tid = omp_get_thread_num();
        if (tid == 0) comm_thread(cfg, cart, report_t, reply_t, log);   /* master = FUNNELED */
        else          port_thread(cfg, tid - 1, cart_rank);
    }   /* implicit barrier: every port thread has returned */

    omp_destroy_lock(&state_lock);
    if (log != stderr) fclose(log);
}

/* ======================= BASE STATION (placeholder for Task 2) =======================
 * receives reports, replies "no nodes available", terminates all nodes after duration.
 * Replace the body of base_run with the real Task 2 implementation. */
void base_run(const Config *cfg, MPI_Datatype report_t, MPI_Datatype reply_t) {
    int num_nodes = cfg->m * cfg->n;
    FILE *log = fopen("base_station.log", "w");
    if (!log) log = stderr;
    fprintf(log, "BASE STUB grid=%dx%d k=%d threshold=%d cycle=%.2fs duration=%.1fs\n",
            cfg->m, cfg->n, cfg->k, cfg->threshold, cfg->cycle, cfg->duration);

    Report rep;
    Reply  reply;
    MPI_Status st;
    MPI_Request req;
    int alerts = 0;
    double start = MPI_Wtime();

    MPI_Irecv(&rep, 1, report_t, MPI_ANY_SOURCE, TAG_REPORT, MPI_COMM_WORLD, &req);
    while (MPI_Wtime() - start < cfg->duration) {
        int flag = 0;
        MPI_Test(&req, &flag, &st);
        if (!flag) {
            struct timespec ts = { 0, 10 * 1000 * 1000 };   /* poll every 10 ms */
            nanosleep(&ts, NULL);
            continue;
        }
        alerts++;
        fprintf(log, "[%.3f] ALERT from node %d (%d,%d) at %04d-%02d-%02d %02d:%02d:%02d avail=%d "
                "nbr_avail=%d/%d/%d/%d msgs=%d delay=%.3f ms\n",
                MPI_Wtime() - start, rep.cart_rank, rep.coords[0], rep.coords[1],
                rep.year, rep.month, rep.day, rep.hour, rep.minute, rep.second, rep.availability,
                rep.nbr_avail[0], rep.nbr_avail[1], rep.nbr_avail[2], rep.nbr_avail[3],
                rep.msgs_with_neighbours, (MPI_Wtime() - rep.send_wtime) * 1e3);

        memset(&reply, 0, sizeof reply);          /* TODO Task 2: nearest available nodes */
        reply.send_wtime = MPI_Wtime();
        MPI_Send(&reply, 1, reply_t, st.MPI_SOURCE, TAG_REPLY, MPI_COMM_WORLD);
        MPI_Irecv(&rep, 1, report_t, MPI_ANY_SOURCE, TAG_REPORT, MPI_COMM_WORLD, &req);
    }

    /* stop listening for reports with the posted receive */
    MPI_Cancel(&req);
    MPI_Wait(&req, &st);
    int cancelled;
    MPI_Test_cancelled(&st, &cancelled);
    if (!cancelled) alerts++;                     /* it matched a late report */

    /* termination message to every node (point-to-point: nodes are not in a
     * collective with us, and the base station is not in the cart comm) */
    MPI_Request treq[num_nodes];
    int term = 1;
    for (int i = 0; i < num_nodes; i++)
        MPI_Isend(&term, 1, MPI_INT, i + 1, TAG_TERM, MPI_COMM_WORLD, &treq[i]);
    MPI_Waitall(num_nodes, treq, MPI_STATUSES_IGNORE);

    /* drain in-flight reports until every node confirms shutdown */
    int done = 0, late = 0;
    while (done < num_nodes) {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
        if (st.MPI_TAG == TAG_DONE) {
            int sent;
            MPI_Recv(&sent, 1, MPI_INT, st.MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            done++;
        } else {
            MPI_Recv(&rep, 1, report_t, st.MPI_SOURCE, TAG_REPORT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            late++;
        }
    }
    fprintf(log, "TERMINATED: sim_time=%.3fs alerts=%d late_reports_drained=%d\n",
            MPI_Wtime() - start, alerts, late);
    if (log != stderr) fclose(log);
}

/* ======================= main ======================= */
int main(int argc, char **argv) {
    int provided, world_rank, world_size;
    /* only the master thread of each process makes MPI calls */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Config cfg;
    cfg.m         = arg_i(argc, argv, 1, 3);
    cfg.n         = arg_i(argc, argv, 2, 3);
    cfg.k         = arg_i(argc, argv, 3, 6);
    cfg.cycle     = arg_d(argc, argv, 4, 1.0);
    cfg.duration  = arg_d(argc, argv, 5, 30.0);
    cfg.threshold = arg_i(argc, argv, 6, 2);
    cfg.busy_pct  = arg_i(argc, argv, 7, 60);
    cfg.variant   = arg_i(argc, argv, 8, VAR_NEIGHBOR_ALLGATHER);

    int bad = 0;
    if (provided < MPI_THREAD_FUNNELED) bad = 1;
    else if (cfg.m < 1 || cfg.n < 1 || cfg.m * cfg.n + 1 != world_size) bad = 2;
    else if (cfg.k < 1 || cfg.k > MAX_PORTS) bad = 3;
    if (bad) {
        if (world_rank == 0) {
            if (bad == 1) fprintf(stderr, "MPI_THREAD_FUNNELED not supported\n");
            if (bad == 2) fprintf(stderr, "need -np %d for a %dx%d grid (got %d)\n",
                                  cfg.m * cfg.n + 1, cfg.m, cfg.n, world_size);
            if (bad == 3) fprintf(stderr, "k must be 1..%d\n", MAX_PORTS);
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
