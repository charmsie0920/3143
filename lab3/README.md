# FIT3143 Lab 3 - Distributed WSN of EV charging nodes
Charmaine Chee Hing Yi 33520399 | Nazara Glaukofani Lopulisa 35135271

**Everything to submit is in `final/`.** The other folders are development history.

| Path | Contents |
|---|---|
| `final/wsn_ver3.c` | final implementation (Task 1 charging nodes + Task 2 base station) |
| `final/wsn_sweep.job`, `final/submit_chain.sh` | CAAS Slurm scripts for the Task 3 sweep |
| `final/runs/` | every CAAS run: `run_*/base_station.log`, `run_*/node_*.log`, `results.csv` |
| `final/graphs/` | Task 3 graphs, `tables/` and `plot_task3.py` |
| `final/verify_replies.py` | replays every base station decision in `runs/` and re-checks it (`python3 verify_replies.py runs`) |
| `task1/`, `task2/` | earlier versions, kept for reference |
| `task3/Dockerfile` | local Open MPI + OpenMP test environment |

Still to add to `final/` before submitting: presentation slides (PDF) and the AI declaration (PDF).

    mpicc -fopenmp -Wall -o wsn wsn_ver3.c
    mpirun -np <m*n+1> ./wsn m n [k cycle duration threshold busy_pct variant period base_variant poll_ms nbr_timeout_ms dead_node out_dir]

On CAAS, from `final/`: `sbatch wsn_sweep.job` (or `bash submit_chain.sh 6`), then `python3 graphs/plot_task3.py runs graphs`.
