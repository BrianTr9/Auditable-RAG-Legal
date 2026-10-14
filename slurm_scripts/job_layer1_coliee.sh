#!/bin/bash
#SBATCH --job-name=AuRAG_L1_COLIEE
#SBATCH --account=a_eecs_ds
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Layer 1 retrieval on COLIEE, one query file per year (H18-R06), same settings as notebook cell 3.
# Systems run sequentially in ONE job: every SPHR run rebuilds evaluation/vector_db_sphr and every
# flat run rebuilds evaluation/vector_db_flat, so two Layer-1 jobs at once would corrupt each other.
# Aggregate on the Mac: python3 evaluation/aggregate_coliee.py layer1 experiment_results/coliee_layer1
set -eo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repo root: cd $AURAG_DIR && sbatch slurm_scripts/<job>.sh}/slurm_scripts/_common.sh"

ROOT=$RESULTS/coliee_layer1
prov "$ROOT"
COMMON="--dataset coliee --top-k 3,5,10 --retrieval-mode hybrid --bm25-weight 0.5 --rrf-k 60 --device cuda --seed 42 $EXTRA"

for YEAR in "${YEARS[@]}"; do
  OUT=$ROOT/$YEAR/layer1_retrieval; mkdir -p "$ROOT/$YEAR"
  Q="--queries benchmark/COLIEE/simple/simple_${YEAR}_jp.xml"
  # evaluate_retrieval.py writes <output stem>_k{3,5,10}.json; k=10 is written last.
  run_if_missing "${OUT}_sphr_k10.json" \
    python evaluation/evaluate_retrieval.py $COMMON $Q --system sphr --chunk-workers 3 --chunk-map-chunksize 16 --output "${OUT}_sphr.json"
  run_if_missing "${OUT}_flat_chunksize300_k10.json" \
    python evaluation/evaluate_retrieval.py $COMMON $Q --system flat --child-chunk-size 300 --child-chunk-overlap 90 --output "${OUT}_flat_chunksize300.json"
  run_if_missing "${OUT}_flat_chunksize512_k10.json" \
    python evaluation/evaluate_retrieval.py $COMMON $Q --system flat --child-chunk-size 512 --child-chunk-overlap 150 --output "${OUT}_flat_chunksize512.json"
done
