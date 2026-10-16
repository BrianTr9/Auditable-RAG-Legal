#!/bin/bash --login
#SBATCH --job-name=AuRAG_L2_GPT4o
#SBATCH --account=a_eecs_ds
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-18%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# GPT-4o baseline (API JSON mode), one array task per year, same settings as notebook cell 4.
# Still needs a GPU: evaluate_generation.py loads the 8B model to build the identical prompt and references.
# Needs outbound internet from the compute node: smoke-test with --sample-queries 5 first.
set -eo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repo root: cd $AURAG_DIR && sbatch slurm_scripts/<job>.sh}/slurm_scripts/_common.sh"

require_gpu_llama
export OPENAI_API_KEY=$(cat ~/.aurag_openai)       # §3.9; never commit the key
# Every year must hit the same snapshot, so real runs pin it (take it from the smoke test's 'api_model'):
#   OPENAI_MODEL=gpt-4o-YYYY-MM-DD sbatch slurm_scripts/job_layer2_gpt4o.sh
# The smoke test (RESULTS set elsewhere) may use the bare alias.
if [[ $RESULTS == experiment_results && -z ${OPENAI_MODEL:-} ]]; then
  echo "Set OPENAI_MODEL to a dated snapshot for the real run (see the smoke test's api_model)." >&2; exit 1
fi
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o}
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}
OUT=$RESULTS/coliee_layer2/$YEAR
mkdir -p "$OUT"

run_if_missing "$OUT/layer2_generation_gpt4o_short.json" \
  python evaluation/evaluate_generation.py --dataset coliee --queries "benchmark/COLIEE/simple/simple_${YEAR}_jp.xml" \
    --llm-model "$LLM" --seed 42 $EXTRA --mode gpt4o --openai-model "$OPENAI_MODEL" --citation-mode short --output "$OUT/layer2_generation_gpt4o_short.json"
