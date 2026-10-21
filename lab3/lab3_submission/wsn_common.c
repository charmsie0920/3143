/* FIT3143 Lab 3 - shared helpers used by both Task 1 (nodes) and Task 2 (base station) */
#include "wsn_common.h"

/* ======================= running statistics ======================= */
void acc_init(Acc *a) { a->n = 0; a->sum = 0; a->min = DBL_MAX; a->max = -DBL_MAX; }

void acc_add(Acc *a, double x) {
    a->n++;
    a->sum += x;
    if (x < a->min) a->min = x;
    if (x > a->max) a->max = x;
}

void acc_merge(Acc *a, const Acc *b) {
    if (b->n == 0) return;
    a->n += b->n;
    a->sum += b->sum;
    if (b->min < a->min) a->min = b->min;
    if (b->max > a->max) a->max = b->max;
}

void node_stats_init(NodeStats *s) {
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

/* ======================= small utilities ======================= */
int arg_i(int argc, char **argv, int i, int def) { return argc > i ? atoi(argv[i]) : def; }
double arg_d(int argc, char **argv, int i, double def) { return argc > i ? atof(argv[i]) : def; }

void sleep_seconds(double s) {
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
double clock_sync_node(void) {
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

void clock_sync_base(int num_nodes) {
    for (int i = 1; i <= num_nodes; i++)
        for (int r = 0; r < SYNC_ROUNDS; r++) {
            double t;
            MPI_Recv(&t, 1, MPI_DOUBLE, i, TAG_SYNC, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            t = MPI_Wtime();
            MPI_Send(&t, 1, MPI_DOUBLE, i, TAG_SYNC, MPI_COMM_WORLD);
        }
}
