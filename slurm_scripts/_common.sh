# Shared setup for the AuRAG Slurm jobs (sourced, not submitted).
# See z_docs/BunyaHPC/BUNYA_AURAG_EXPERIMENT_WORKFLOW.md §3 for the one-time install.

# ~/.bashrc and conda's activate scripts touch unset variables, so `set -u` comes after them.
source ~/.bashrc                                    # AURAG_DIR, HF_HOME, PIP_CACHE_DIR (§3.4)
module load miniforge
# CUDA modules exist only on GPU nodes; L1 and GPT-4o run on CPU nodes (partition general).
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then module load cuda; fi
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"      # as in RCC's conda guide
conda activate aurag
set -u
cd "${AURAG_DIR:-$SLURM_SUBMIT_DIR}"
# CPU libraries (torch, BLAS) would otherwise size their thread pools to the whole node.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# tiktoken caches its encoding in $TMPDIR by default, which Bunya creates per job and deletes at the end,
# so every job would download it again. Keep it in the cache filled once during setup (workflow §3.7).
export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/scratch/user/$USER/.cache/tiktoken}
# Do NOT set HF_HUB_OFFLINE=1: transformers 4.57.x queries the Hub API (model_info) while loading a tokenizer
# by repo name and raises in offline mode even when the files are cached.

# Fail fast if llama-cpp-python was built without CUDA: it would silently decode on the CPU (workflow §3.6).
require_gpu_llama() {
  python -c "import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 'llama-cpp-python has no GPU support: rebuild it (workflow §3.6)')"
}

# Years to run (array task i -> YEARS[i]). A smoke test may pass a subset, e.g. YEARS_LIST=R06 with --array=0.
read -r -a YEARS <<< "${YEARS_LIST:-H18 H19 H20 H21 H22 H23 H24 H25 H26 H27 H28 H29 H30 R01 R02 R03 R04 R05 R06}"
LLM=models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
# Smoke test without touching the real results, e.g.:
#   RESULTS=experiment_results/smoke EXTRA="--sample-queries 5" YEARS_LIST=R06 sbatch --qos=debug --time=00:30:00 --array=0 slurm_scripts/job_layer2_coliee.sh
RESULTS=${RESULTS:-experiment_results}
EXTRA=${EXTRA:-}

# Provenance next to the results: commit, GPU, date, packages.
prov() {
  mkdir -p "$1"
  { echo "job=${SLURM_JOB_ID:-} array_task=${SLURM_ARRAY_TASK_ID:-}"; git rev-parse HEAD; { git status --porcelain | head -20 || true; }
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU (CPU job on $(hostname))"
    date -u; } > "$1/RUN_INFO.txt"
  pip freeze > "$1/pip_freeze.txt"
}

# run_if_missing <output file> <command...>: skip work a previous attempt already finished,
# so a failed or timed-out job can simply be resubmitted. Outputs are written only at the end of a run.
run_if_missing() {
  local out=$1; shift
  if [[ -s $out ]]; then echo "skip (exists): $out"; else "$@"; fi
}
