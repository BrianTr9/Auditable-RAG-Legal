#!/bin/bash
# Submit the whole COLIEE H18-R06 re-run in dependency order, from the login node (repo root):
#   bash slurm_scripts/submit_all.sh
#   1. setup (CPU node, short queue): conda env, packages, model, e5, tiktoken
#   2. after 1 succeeds: Layer 1 (CPU), GPT-4o (CPU) and the GPU job (CUDA build, L2 smoke, L2 for all years)
# Every job only creates files under the repo, ~/.conda and /scratch caches; nothing is deleted.
# If setup fails, the dependent jobs never start (squeue shows DependencyNeverSatisfied): read
# logs/aurag_setup_cpu_<id>.out, cancel with `scancel -u $USER`, fix, and run this script again.
set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

SETUP=$(sbatch --parsable --job-name=aurag_setup_cpu --account=a_eecs_ds --partition=general --qos=normal \
  --constraint=epyc3 --cpus-per-task=8 --mem=32G --time=02:00:00 --output=logs/%x_%j.out \
  --wrap="bash -l slurm_scripts/setup_env.sh cpu")
L1=$(sbatch --parsable --dependency=afterok:$SETUP slurm_scripts/job_layer1_coliee.sh)
GPT=$(sbatch --parsable --dependency=afterok:$SETUP slurm_scripts/job_layer2_gpt4o.sh)
GPU=$(sbatch --parsable --dependency=afterok:$SETUP slurm_scripts/job_gpu_all.sh)

echo "setup=$SETUP  layer1=$L1  gpt4o=$GPT  gpu_all=$GPU"
echo "Watch: squeue --me      Logs: ls logs/"
