#!/bin/bash
# Smoke test of the CPU jobs from the login node (after `setup_env.sh cpu` finished):
#   bash slurm_scripts/smoke_test.sh          submit L1 and GPT-4o: R06 only, 5 sampled queries, debug QoS
#   bash slurm_scripts/smoke_test.sh report   after they finish: outputs, errors, threads, GPT-4o snapshot
# The Layer-2 (GPU) smoke test runs at the end of `setup_env.sh` in GPU mode.
# The smoke settings live only inside this script, so they cannot leak into a real run.
cd "$(dirname "$0")/.."

if [[ ${1:-} == report ]]; then
  R=experiment_results/smoke
  echo "== setup logs";          tail -n 2 logs/aurag_setup*.out 2>/dev/null
  echo "== output files";        find $R -name '*.json' 2>/dev/null | sort
  echo "== stderr tails";        tail -n 3 logs/AuRAG_*.err 2>/dev/null
  echo "== llama.cpp threads";   grep -h "Threads:" logs/AuRAG_L2*.out logs/aurag_setup*.out 2>/dev/null | sort -u
  echo "== GPT-4o snapshot";     grep -ho '"api_model": "[^"]*"' $R/coliee_layer2/R06/layer2_generation_gpt4o_short_raw_outputs.json 2>/dev/null | sort | uniq -c
  exit 0
fi

set -eo pipefail
mkdir -p logs
export RESULTS=experiment_results/smoke EXTRA="--sample-queries 5" YEARS_LIST=R06
sbatch --qos=debug --time=00:30:00 --array=0 slurm_scripts/job_layer2_gpt4o.sh
sbatch --qos=debug --time=00:45:00 slurm_scripts/job_layer1_coliee.sh
echo "Submitted. Watch with: squeue --me"
