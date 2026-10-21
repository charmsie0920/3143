/* FIT3143 Lab 3 - TASK 2: base station
 * Charmaine Chee Hing Yi 33520399 | Nazara Glaukofani Lopulisa 35135271
 *
 * Single MPI process (world rank 0), outside the Cartesian grid.
 *
 * FINAL approach : base_variant 1, pre-posted MPI_Irecv + MPI_Test, MPI_Isend reply.
 * ALTERNATIVES   : kept for the Task 3 comparison only:
 *                  0 MPI_Iprobe + MPI_Recv, blocking MPI_Send reply
 *                  2 persistent MPI_Recv_init per node + MPI_Testsome, MPI_Isend reply */
#include "wsn_common.h"

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
