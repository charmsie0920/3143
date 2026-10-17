/* FIT3143 Lab 3 - MINIMAL base station used to test Task 1.
 * PLACEHOLDER for Task 2: receives reports, replies "no nodes available",
 * terminates all nodes after cfg->duration. Replace with base.c (Task 2). */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "types.h"

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
