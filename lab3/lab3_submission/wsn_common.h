/* FIT3143 Lab 3 - Distributed WSN of EV charging nodes (Open MPI + OpenMP)
 * Charmaine Chee Hing Yi 33520399 | Nazara Glaukofani Lopulisa 35135271
 *
 * Source files
 *   wsn_common.h / wsn_common.c  shared message formats, tags, datatypes, clock sync
 *   task1_node.c                 TASK 1: EV charging node (port threads + comm thread)
 *   task2_base.c                 TASK 2: base station
 *   wsn_main.c                   MPI start-up: split base station off, build Cartesian grid
 *
 * compile: make            (or: mpicc -fopenmp -Wall -o wsn wsn_main.c wsn_common.c task1_node.c task2_base.c)
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
 *                    0 Iprobe+Recv, blocking Send reply
 *                    1 Irecv+Test (pre-posted receive), Isend reply (default)
 *                    2 persistent Recv_init per node + Testsome, Isend reply
 *   poll_ms        how often idle processes check for messages (ms)
 *   nbr_timeout_ms on-demand: give up waiting for a neighbour after this long
 *   dead_node      cart rank that stops answering its neighbours (-1 = none, on-demand only)
 *   out_dir        results folder: <out_dir>/run_<m>x<n>_v<variant>_b<base>_<time>/ + results.csv
 *
 * world rank 0 = base station; world ranks 1..m*n = charging nodes (cart rank = world rank - 1) */
#ifndef WSN_COMMON_H
#define WSN_COMMON_H

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
static const char *const dir_name[NUM_DIRS] = { "up", "down", "left", "right" };

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
static const char *const variant_name[NUM_VARIANTS] = { "on-demand", "periodic Neighbor_allgather",
                                                  "periodic Ineighbor_allgather", "periodic Isend/Irecv" };

/* base station listener implementations compared in Task 3 */
enum { BV_IPROBE_RECV = 0,           /* Iprobe + blocking Recv, blocking Send reply     */
       BV_IRECV_TEST = 1,            /* one Irecv(ANY_SOURCE) + Test, Isend reply       */
       BV_PERSISTENT_TESTSOME = 2,   /* Recv_init per node + Testsome, Isend reply      */
       NUM_BV = 3 };
static const char *const bv_name[NUM_BV] = { "Iprobe+Recv/Send", "Irecv+Test/Isend", "Recv_init+Testsome/Isend" };

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

/* ---------- shared helpers (wsn_common.c) ---------- */
void acc_init(Acc *a);
void acc_add(Acc *a, double x);
void acc_merge(Acc *a, const Acc *b);
void node_stats_init(NodeStats *s);
MPI_Datatype build_report_type(void);
MPI_Datatype build_reply_type(void);
int    arg_i(int argc, char **argv, int i, int def);
double arg_d(int argc, char **argv, int i, double def);
void   sleep_seconds(double s);
double clock_sync_node(void);
void   clock_sync_base(int num_nodes);

/* ---------- Task 1: EV charging node (task1_node.c) ---------- */
void node_run(const Config *cfg, MPI_Comm cart, MPI_Datatype report_t, MPI_Datatype reply_t,
              double clock_offset);

/* ---------- Task 2: base station (task2_base.c) ---------- */
void base_run(const Config *cfg, MPI_Datatype report_t, MPI_Datatype reply_t);

#endif
