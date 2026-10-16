#!/bin/bash
# Smoke test from the login node (repo root or anywhere in the repo):
#   bash slurm_scripts/smoke_test.sh          submit the 3 jobs: R06 only, 5 sampled queries, debug QoS
#   bash slurm_scripts/smoke_test.sh report   after they finish: outputs, errors, threads, GPT-4o snapshot
# The smoke settings live only inside this script, so they cannot leak into a real run.
cd "$(dirname "$0")/.."

if [[ ${1:-} == report ]]; then
  R=experiment_results/smoke
  echo "== output files";        find $R -name '*.json' 2>/dev/null | sort
  echo "== stderr tails";        tail -n 3 logs/AuRAG_*.err 2>/dev/null
  echo "== llama.cpp threads";   grep -h "Threads:" logs/AuRAG_L2*.out 2>/dev/null | sort -u
  echo "== GPT-4o snapshot";     grep -ho '"api_model": "[^"]*"' $R/coliee_layer2/R06/layer2_generation_gpt4o_short_raw_outputs.json 2>/dev/null | sort | uniq -c
  exit 0
fi

set -eo pipefail
mkdir -p logs
export RESULTS=experiment_results/smoke EXTRA="--sample-queries 5" YEARS_LIST=R06
sbatch --qos=debug --time=00:30:00 --array=0 slurm_scripts/job_layer2_coliee.sh
sbatch --qos=debug --time=00:30:00 --array=0 slurm_scripts/job_layer2_gpt4o.sh
sbatch --qos=debug --time=00:45:00 slurm_scripts/job_layer1_coliee.sh
echo "Submitted. Watch with: squeue --me   (debug QoS runs 2 jobs at a time; the third waits)"
