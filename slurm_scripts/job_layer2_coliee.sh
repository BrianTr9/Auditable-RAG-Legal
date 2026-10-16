#!/bin/bash --login
#SBATCH --job-name=AuRAG_L2_COLIEE
#SBATCH --account=a_eecs_ds
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-18%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Layer 2 (ideal retrieval, R = G) with the local 8B model, one array task per year (H18-R06),
# same settings as notebook cell 4. Layer 2 builds no index, so tasks can run in parallel.
# Every year, R06 included, is re-run on the same GPU type so latencies are comparable.
# Queries without GT articles (H29-28-E and six in R04) are skipped by evaluate_generation.py.
# Aggregate on the Mac: python3 evaluation/aggregate_coliee.py layer2 experiment_results/coliee_layer2 --compare experiment_results_coliee
set -eo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repo root: cd $AURAG_DIR && sbatch slurm_scripts/<job>.sh}/slurm_scripts/_common.sh"

require_gpu_llama
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}
OUT=$RESULTS/coliee_layer2/$YEAR
prov "$OUT"
GEN="--dataset coliee --queries benchmark/COLIEE/simple/simple_${YEAR}_jp.xml --llm-model $LLM --n-ctx 16384 --max-tokens 2048 --seed 42 $EXTRA"

run_if_missing "$OUT/layer2_generation_rdg.json" \
  python evaluation/evaluate_generation.py $GEN --mode rdg --output "$OUT/layer2_generation_rdg.json"
run_if_missing "$OUT/layer2_generation_prompt_short.json" \
  python evaluation/evaluate_generation.py $GEN --mode prompt --citation-mode short --output "$OUT/layer2_generation_prompt_short.json"
# Lenient row of Table 2: the same 'short' outputs scored with citation_mode='both' (no second generation).
run_if_missing "$OUT/layer2_generation_prompt_both.json" \
  python evaluation/rescore_citations.py --dataset coliee --results "$OUT/layer2_generation_prompt_short.json" \
    --citation-mode both --output "$OUT/layer2_generation_prompt_both.json"
