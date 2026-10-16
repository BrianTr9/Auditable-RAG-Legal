#!/bin/bash --login
#SBATCH --job-name=AuRAG_L2_GPT4o
#SBATCH --account=a_eecs_ds
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-18%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# GPT-4o baseline (API JSON mode), one array task per year, same settings as notebook cell 6.
# CPU only: the prompt comes from PromptBuilder (same function as the 8B runs), no local model is loaded.
# Needs outbound internet from the compute node.
set -eo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repo root: cd $AURAG_DIR && sbatch slurm_scripts/<job>.sh}/slurm_scripts/_common.sh"

# `export V=$(cat f)` would hide a missing file (export returns 0), so test first. Never commit the key.
[[ -s ~/.aurag_openai ]] || { echo "~/.aurag_openai is missing or empty (see workflow doc section 3)" >&2; exit 1; }
OPENAI_API_KEY=$(cat ~/.aurag_openai)
export OPENAI_API_KEY
# Every year must hit the same snapshot. The bare alias "gpt-4o" answered as gpt-4o-2024-08-06 on 2026-10-16.
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-2024-08-06}
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}
OUT=$RESULTS/coliee_layer2/$YEAR
mkdir -p "$OUT"
stamp_results "$RESULTS/coliee_layer2" l2_gpt4o "--dataset coliee --seed 42 $EXTRA|gpt4o|$OPENAI_MODEL|short" \
  '*/layer2_generation_gpt4o_short*.json'

run_if_missing "$OUT/layer2_generation_gpt4o_short.json" \
  python evaluation/evaluate_generation.py --dataset coliee --queries "benchmark/COLIEE/simple/simple_${YEAR}_jp.xml" \
    --llm-model "$LLM" --seed 42 $EXTRA --mode gpt4o --openai-model "$OPENAI_MODEL" --citation-mode short --output "$OUT/layer2_generation_gpt4o_short.json"
