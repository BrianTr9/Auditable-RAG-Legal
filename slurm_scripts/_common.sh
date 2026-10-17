# Shared setup for the AuRAG Slurm jobs (sourced, not submitted).
# See z_docs/BunyaHPC/BUNYA_AURAG_EXPERIMENT_WORKFLOW.md §3 for the one-time install.

# ~/.bashrc and conda's activate scripts touch unset variables, so `set -u` comes after them.
source ~/.bashrc                                    # optional AURAG_DIR / HF_HOME; both fall back to defaults
source "$(dirname "${BASH_SOURCE[0]}")/_modules.sh"
load_module miniforge
# CUDA modules exist only on GPU nodes; L1 and GPT-4o run on CPU nodes (partition general).
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then load_module cuda; fi
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"      # as in RCC's conda guide
conda activate aurag
# A nested script (the smoke test that setup_env.sh starts with `bash --login`) inherits an already active env.
# `conda activate` of the active env leaves PATH untouched, and the module loads above can put the base miniforge
# python in front of it, so `python` would miss every package installed in `aurag`. Put the env first explicitly.
export PATH="${CONDA_PREFIX:?conda activate aurag failed}/bin:$PATH"
hash -r
if [[ $(command -v python) != "$CONDA_PREFIX/bin/python" ]]; then
  echo "ERROR: python is $(command -v python), expected $CONDA_PREFIX/bin/python (is the aurag env complete?)" >&2
  exit 1
fi
set -u
cd "${AURAG_DIR:-$SLURM_SUBMIT_DIR}"
# CPU libraries (torch, BLAS) would otherwise size their thread pools to the whole node.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# tiktoken caches its encoding in $TMPDIR by default, which Bunya creates per job and deletes at the end,
# so every job would download it again. Keep it in the cache filled once during setup (setup_env.sh).
export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/scratch/user/$USER/.cache/tiktoken}
# Do NOT set HF_HUB_OFFLINE=1: transformers 4.57.x queries the Hub API (model_info) while loading a tokenizer
# by repo name and raises in offline mode even when the files are cached.

# Fail fast if llama-cpp-python was built without CUDA: it would silently decode on the CPU (setup_env.sh gpu mode builds it with CUDA).
require_gpu_llama() {
  python -c "import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 'llama-cpp-python has no GPU support: rebuild it (setup_env.sh, gpu mode)')"
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
    date -u; echo "---"; } >> "$1/RUN_INFO.txt"   # appended: one block per job, so earlier runs stay on record
  pip freeze > "$1/pip_freeze.txt"
}

# ---- result stamps: never reuse results made by different code or settings --------------------------------
# Fingerprint of everything that can change a number: Python sources, the COLIEE corpus and query files.
# Left out because they never produce a number: Slurm scripts, docs, tests, aggregate_coliee.py (it only reads the
# result JSON, on the Mac) and evaluation/systems/ (end-to-end only). Editing them must not invalidate finished results.
# Anything else under src/ and evaluation/ counts, so a new file is covered by default.
code_fingerprint() {
  { find src evaluation -name '*.py' -not -path '*/__pycache__/*' -not -path 'evaluation/systems/*' -not -name 'aggregate_coliee.py'
    ls benchmark/COLIEE/civil.xml benchmark/COLIEE/simple/*.xml; } \
    | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1
}

# stamp_results <root> <label> <settings> <glob relative to root>...
# Call once per job, before any run_if_missing. The stamp (hash of code and settings) is kept in <root>/.stamp_<label>:
#   - same stamp: finished results are reused;
#   - different stamp: this label's old results move to <root>/../_stale/<label>/<UTC time>_<old stamp>/ (kept, never
#     deleted) and are recomputed, so a table never mixes code versions;
#   - no stamp but results exist (made before stamps): adopted under the current stamp, with a warning.
# Safe for parallel array tasks (flock); each label only touches its own files.
stamp_results() {
  local root=$1 label=$2 settings=$3; shift 3
  CODE_FP=$(code_fingerprint)
  local stamp
  stamp=$(printf '%s\n%s\n' "$settings" "$CODE_FP" | sha256sum | cut -d' ' -f1)
  mkdir -p "$root"
  (
    flock 9
    cd "$root"
    shopt -s nullglob
    local files=() pat f old="" dest
    for pat in "$@"; do for f in $pat; do [[ -f $f ]] && files+=("$f"); done; done
    if [[ -s .stamp_$label ]]; then old=$(<".stamp_$label"); fi
    if [[ -n $old && $old != "$stamp" && ${#files[@]} -gt 0 ]]; then
      dest="$(cd .. && pwd)/_stale/$label/$(date -u +%Y%m%dT%H%M%SZ)_${old:0:8}"
      echo "STAMP $label: code or settings changed since these results were made; moving ${#files[@]} old files to $dest"
      for f in "${files[@]}"; do mkdir -p "$dest/$(dirname "$f")"; mv "$f" "$dest/$f"; done
    elif [[ -z $old && ${#files[@]} -gt 0 ]]; then
      echo "STAMP $label: ${#files[@]} result files have no stamp (made before stamps existed); adopting them for the current code and settings"
    fi
    echo "$stamp" > ".stamp_$label"
  ) 9>"$root/.stamp_$label.lock"
  echo "stamp $label: ${stamp:0:12}"
}

# run_if_missing <output file> <command...>: skip work a previous attempt already finished,
# so a failed or timed-out job can simply be resubmitted. Outputs are written only at the end of a run.
# After stamp_results it also stops the job if the code changed since the stamp (e.g. a `git pull` mid-job), because
# the remaining years would otherwise run on different code than the finished ones.
run_if_missing() {
  local out=$1; shift
  if [[ -n ${CODE_FP:-} && $(code_fingerprint) != "$CODE_FP" ]]; then
    echo "ERROR: src/, evaluation/ or the COLIEE data changed while this job was running; resubmit it." >&2
    exit 1
  fi
  if [[ -s $out ]]; then echo "skip (exists): $out"; else "$@"; fi
}
