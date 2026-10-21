/* FIT3143 Lab 3 - TASK 1: EV charging node
 * Charmaine Chee Hing Yi 33520399 | Nazara Glaukofani Lopulisa 35135271
 *
 * One MPI process per node in the m x n Cartesian grid. k port threads update a shared
 * array; thread 0 (comm thread) does all MPI calls (MPI_THREAD_FUNNELED).
 *
 * FINAL approach : variant 0, ON-DEMAND - ask neighbours only when congested
 *                  (start_round / service_neighbours / finish_round).
 * ALTERNATIVES   : variants 1-3, PERIODIC exchange every cycle, kept for the Task 3
 *                  comparison only (periodic_exchange): 1 MPI_Neighbor_allgather,
 *                  2 MPI_Ineighbor_allgather, 3 MPI_Isend/MPI_Irecv + Waitall. */
#include "wsn_common.h"

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

