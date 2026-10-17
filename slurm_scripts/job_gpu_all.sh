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
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Everything that needs a GPU, in ONE allocation (GPUs on Bunya can queue for hours to days, so avoid queueing twice):
#   1. setup_env.sh in GPU mode: build llama-cpp-python with CUDA, then a 2-query Layer-2 smoke test;
#      it exits non-zero on any problem, which stops this job before the real run.
#   2. Layer 2 (RDG, prompt-only, Lenient re-scoring) for all 19 years, one after another.
# Resumable: finished outputs are skipped, so after a timeout just submit this job again.
# Measured: 2 h 49 min on one A100 80GB for all 19 years (1216 queries). The 5 h request leaves ~1.8x margin and is
# scheduled sooner than a longer one; the first run also builds llama-cpp with CUDA (~20 min), later runs skip it.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
# sbatch exports the submitting shell's environment. This job is always the real run, so a RESULTS, EXTRA or YEARS_LIST
# left over there (they select the smoke run) must not leak in; setup_env.sh sets them explicitly for its own smoke test.
unset RESULTS EXTRA YEARS_LIST

bash --login slurm_scripts/setup_env.sh

# One failing year must not stop the others: a resubmit would queue for a GPU again (days).
# Failed years are listed at the end (task i = i-th year of YEARS in _common.sh) and the job exits non-zero.
FAILED=()
for i in $(seq 0 18); do
  if ! SLURM_ARRAY_TASK_ID=$i bash --login slurm_scripts/job_layer2_coliee.sh; then
    echo "LAYER 2 FAILED for task $i, continuing with the next year" >&2
    FAILED+=("$i")
  fi
done
if (( ${#FAILED[@]} > 0 )); then
  echo "LAYER 2 INCOMPLETE: failed tasks: ${FAILED[*]}. Fix, then submit this job again (finished outputs are skipped)." >&2
  exit 1
fi
echo "LAYER 2 DONE: experiment_results/coliee_layer2/"
