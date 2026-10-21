#!/bin/bash
# Submit wsn_sweep.job N times, each job waiting for the previous one to end
# (afterany), so the whole sweep runs back to back within the one-running-job
# limit without resubmitting by hand. Extra sbatch options are passed through.
#
#   bash submit_chain.sh 6
#   bash submit_chain.sh 3 --export=ALL,SUITE=line
N=${1:-6}
shift
prev=$(sbatch --parsable "$@" wsn_sweep.job) || exit 1
echo "submitted $prev"
for i in $(seq 2 "$N"); do
    prev=$(sbatch --parsable --dependency=afterany:"$prev" "$@" wsn_sweep.job) || exit 1
    echo "submitted $prev (after the previous job)"
done
echo "check with: squeue -u \$USER"
