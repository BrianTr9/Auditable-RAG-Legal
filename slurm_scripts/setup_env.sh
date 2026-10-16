#!/bin/bash
# One-time setup on a Bunya GPU compute node: run inside `salloc` on the GPU type the jobs use
# (a100), from anywhere in the repo:  bash slurm_scripts/setup_env.sh
# Safe to re-run: existing env, packages and downloads are kept. Deletes nothing and does not
# touch ~/.bashrc. The conda env lives in ~/.conda (RCC policy); caches go to /scratch.
set -eo pipefail
cd "$(dirname "$0")/.."

module load miniforge cuda
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"
conda env list | grep -q "^aurag " || conda create -n aurag python=3.10 -y
conda activate aurag

export PIP_CACHE_DIR=/scratch/user/$USER/.cache/pip
export TIKTOKEN_CACHE_DIR=/scratch/user/$USER/.cache/tiktoken   # same default as _common.sh
pip install --upgrade pip setuptools wheel cmake

# llama-cpp-python with CUDA first, so requirements.txt finds it installed and never builds a CPU-only copy.
if ! python -c "import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)" 2>/dev/null; then
  pip uninstall -y llama-cpp-python 2>/dev/null || true
  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80;89;90" pip install llama-cpp-python --no-cache-dir
fi
pip install -r requirements.txt scipy

# Downloads (also a check that compute nodes reach the internet).
mkdir -p models logs
wget -c -O models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

python -c "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload(), 'no GPU support'"
echo "SETUP OK: env aurag, llama-cpp with GPU, model, e5 and tiktoken ready."
