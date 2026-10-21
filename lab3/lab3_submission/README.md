# FIT3143 Lab 3: Distributed WSN of EV charging nodes

Charmaine Chee Hing Yi 33520399 · Nazara Glaukofani Lopulisa 35135271

## Source files

| File | What it is |
|---|---|
| `task1_node.c` | **Task 1**: EV charging node (k port threads + 1 communication thread, neighbour requests, alerts, graceful shutdown) |
| `task2_base.c` | **Task 2**: base station (listens for reports, finds the nearest available nodes, logging, termination) |
| `wsn_common.h`, `wsn_common.c` | Helpers shared by both tasks: message tags, `Report`/`Reply` structs and their MPI derived datatypes, clock synchronisation |
| `wsn_main.c` | Start-up: splits the base station off `MPI_COMM_WORLD`, builds the m × n Cartesian grid, then calls `node_run()` (Task 1) or `base_run()` (Task 2) |
| `Makefile` | Builds everything into one executable, `wsn` |

Tasks 1 and 2 run as one MPI program (one `mpirun`), so both tasks are compiled into the same executable. World rank 0 runs Task 2 and every other rank runs Task 1.

## Final version and alternative approaches

The alternatives we compared in Task 3 are kept in the same files and selected with a command-line argument. The defaults are the final versions.

| | Final (default) | Alternatives (Task 3 comparison only) |
|---|---|---|
| **Task 1**: how a node gets its neighbours' availability (`variant`, argument 8) | **0: on-demand**: `MPI_Isend` a request only when congested, receive with `MPI_Iprobe` → `MPI_Recv` (`start_round`, `service_neighbours`, `finish_round`) | 1: periodic `MPI_Neighbor_allgather` every cycle<br>2: periodic `MPI_Ineighbor_allgather`<br>3: periodic `MPI_Isend`/`MPI_Irecv` + `MPI_Waitall`<br>(all in `periodic_exchange`) |
| **Task 2**: how the base station listens for reports (`base_variant`, argument 10) | **1: pre-posted `MPI_Irecv` + `MPI_Test`**, reply with `MPI_Isend` | 0: `MPI_Iprobe` + `MPI_Recv`, blocking `MPI_Send` reply<br>2: persistent `MPI_Recv_init` per node + `MPI_Testsome` |

## Build and run

```bash
make                                   # or: mpicc -fopenmp -Wall -o wsn wsn_main.c wsn_common.c task1_node.c task2_base.c
mpirun -np <m*n+1> ./wsn m n [k cycle duration threshold busy_pct variant period base_variant poll_ms nbr_timeout_ms dead_node out_dir]
```

Examples:

```bash
mpirun -np 10 ./wsn 3 3                              # 3x3 grid, final versions, default settings
mpirun -np 10 ./wsn 3 3 6 1.0 30 2 90                # low-availability test (90% of ports busy)
mpirun -np 10 ./wsn 3 3 6 1.0 30 2 85 0 5.0 1 1.0 300 4   # node 4 stops answering (dead-node test)
mpirun -np 10 ./wsn 3 3 6 1.0 30 2 85 1 5.0 0        # alternative: periodic Neighbor_allgather + Iprobe/Recv base
```

Defaults: k = 6 ports, 1 s cycle, 30 s run, threshold 2, 60% busy, variant 0, period 5 s, base_variant 1, poll 1 ms, neighbour timeout 500 ms, no dead node, output folder `logs`.

Output: `<out_dir>/run_<m>x<n>_v<variant>_b<base_variant>_<time>/base_station.log` and `node_<id>.log`, plus one row per message type in `<out_dir>/results.csv`.

## Task 3 files (`task3/`)

| File | What it is |
|---|---|
| `wsn_sweep.job` | Resumable Slurm job: compiles the code and runs every configuration ("suite"), one folder per run, each run split over 2 CAAS compute nodes |
| `submit_chain.sh` | Queues `wsn_sweep.job` several times back to back (10-minute job limit) |
| `plot_task3.py` | Reads the runs and draws every graph and table: `python3 plot_task3.py runs graphs` |
| `graphs/` | The graphs used in the presentation, plus `tables/` with every number behind them |
| `runs/` | All 208 CAAS runs: `base_station.log`, every `node_<id>.log`, `results.csv` per run |

Folder names in `runs/` say what each run was: `<suite>_<m>x<n>_v<variant>_b<base_variant>_p<period>_busy<pct>_r1`.
The Task 3a/3b graphs use the final versions (`v0_b1`); runs with `b0` were measured before we switched the
base station to `Irecv` + `Test` and are kept for the comparison.

Reproduce on CAAS (put `wsn_sweep.job` and `submit_chain.sh` in the same folder as the `.c` files, since the job compiles them):

```bash
bash submit_chain.sh 11 --export="ALL,SUITE=grid line,DURATION=60"   # Task 3a (32 grid shapes) + 3b (30 line lengths)
bash submit_chain.sh 2 --export="ALL,SUITE=period demo,DURATION=20"  # predefined-period graph + demo runs
bash submit_chain.sh 4 --export="ALL,SUITE=variants,DURATION=20"     # node and base station version comparison
python3 plot_task3.py runs graphs
```
