/* FIT3143 Lab 3 - program start-up shared by Task 1 and Task 2
 * Splits the base station (world rank 0) off, builds the m x n Cartesian grid for the
 * charging nodes, then runs base_run() (task2_base.c) or node_run() (task1_node.c). */
#include "wsn_common.h"

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
    cfg.base_variant   = arg_i(argc, argv, 10, BV_IRECV_TEST);
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
