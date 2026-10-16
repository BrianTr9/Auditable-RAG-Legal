#!/bin/bash --login
#SBATCH --job-name=AuRAG_GPU_ALL
#SBATCH --account=a_eecs_ds
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Everything that needs a GPU, in ONE allocation (GPUs on Bunya queue for days, so avoid queueing twice):
#   1. setup_env.sh in GPU mode: build llama-cpp-python with CUDA, then a 2-query Layer-2 smoke test;
#      it exits non-zero on any problem, which stops this job before the real run.
#   2. Layer 2 (RDG, prompt-only, Lenient re-scoring) for all 19 years, one after another.
# Resumable: finished outputs are skipped, so after a timeout just submit this job again.
# Estimate: ~6 h on one A100 (1216 queries x ~14 s, plus model loads and the CUDA build).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

bash --login slurm_scripts/setup_env.sh
for i in $(seq 0 18); do
  SLURM_ARRAY_TASK_ID=$i bash --login slurm_scripts/job_layer2_coliee.sh
done
echo "LAYER 2 DONE: experiment_results/coliee_layer2/"
