#!/bin/bash
# One-time setup on a Bunya compute node, in two modes:
#   bash -l slurm_scripts/setup_env.sh cpu   CPU node (partition general, short queue): conda env,
#                                            Python packages and all downloads; no llama-cpp.
#   bash -l slurm_scripts/setup_env.sh       GPU node (default mode): additionally builds llama-cpp-python
#                                            with CUDA, then smoke-tests Layer 2 (R06, 2 queries) with the
#                                            real job script, so the GPU path is checked in this allocation.
# Safe to re-run in either order: existing env, packages and downloads are kept. Deletes nothing and
# does not touch ~/.bashrc. The conda env lives in ~/.conda (RCC policy); caches go to /scratch.
set -eo pipefail
MODE=${1:-gpu}
cd "$(dirname "$0")/.."

# Diagnostics first, so a failure is easy to place in the log.
echo "setup_env.sh mode=$MODE host=$(hostname) date=$(date -u) gpus=${CUDA_VISIBLE_DEVICES:-none}"
echo "internet (HTTP code from huggingface.co): $(curl -sS --max-time 15 -o /dev/null -w '%{http_code}' https://huggingface.co 2>&1 || true)"
set -x   # echo every command from here on

source slurm_scripts/_modules.sh
load_module miniforge
if [[ $MODE == gpu ]]; then load_module cuda; fi
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
# Match against a variable, not a pipe: under pipefail an early-exiting `grep -q` can SIGPIPE `conda env list`,
# the pipeline then reports failure and `conda create -y` would be run on the existing env.
ENVS=$(conda env list)
grep -q "^aurag " <<< "$ENVS" || conda create -n aurag python=3.10 -y
conda activate aurag

export PIP_CACHE_DIR=/scratch/user/$USER/.cache/pip
export TIKTOKEN_CACHE_DIR=/scratch/user/$USER/.cache/tiktoken   # same default as _common.sh
pip install --upgrade pip setuptools wheel cmake

if [[ $MODE == gpu ]]; then
  # llama-cpp-python with CUDA first, so requirements.txt finds it installed and never builds a CPU-only copy.
  if ! python -c "import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)" 2>/dev/null; then
    pip uninstall -y llama-cpp-python 2>/dev/null || true
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80;89;90" pip install llama-cpp-python --no-cache-dir
  fi
  pip install -r requirements.txt scipy
else
  # CPU mode: everything except llama-cpp-python (L1 and GPT-4o do not use it; the GPU mode builds it).
  pip install -r <(grep -v '^llama-cpp-python' requirements.txt) scipy
fi

# Downloads (also a check that compute nodes reach the internet).
mkdir -p models logs
wget -c -O models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

if [[ $MODE == cpu ]]; then
  echo "SETUP OK (cpu): env aurag, packages, model, e5 and tiktoken ready. GPU mode still needed for llama-cpp."
  exit 0
fi

python -c "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload(), 'no GPU support'"
echo "SETUP OK (gpu): llama-cpp with GPU ready."

# Layer-2 smoke test inside this GPU allocation, through the real job script (results kept apart).
RESULTS=experiment_results/smoke EXTRA="--sample-queries 2" YEARS_LIST=R06 SLURM_ARRAY_TASK_ID=0 \
  SLURM_SUBMIT_DIR="$PWD" bash --login slurm_scripts/job_layer2_coliee.sh
echo "L2 SMOKE OK: see experiment_results/smoke/coliee_layer2/R06/"
