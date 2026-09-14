#!/bin/bash
# collect_results.sh
#
# Pulls every "CSV," line out of the slurm output files into one results.csv.
# All four implementations emit the same column layout, so a single file
# holds serial, pthreads, OpenMP and MPI results together.
#
# Safe to re-run: it rebuilds from scratch, so running it again after more
# jobs finish simply picks up the new results as well.
#
# Usage: bash collect_results.sh

OUT=results.csv

# write_s is the file-write time. Runs from before it was added have no value
# there; analyse.py flags them, since their totals exclude the write.
echo "impl,scheme,n,procs,threads,nodes,primes,total_s,serial_s,parallel_s,overhead_s,imbalance_pct,write_s" > $OUT

cat slurm-*.out 2>/dev/null | grep '^CSV,' | sed 's/^CSV,//' >> $OUT

RUNS=$(($(wc -l < $OUT) - 1))
echo "Wrote $OUT with $RUNS runs."
echo
echo "Runs per configuration:"
tail -n +2 $OUT | awk -F, '{print $1, $2, "n="$3, "p="$4, "th="$5, "nodes="$6}' \
    | sort | uniq -c | sort -k2