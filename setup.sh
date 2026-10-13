#!/bin/bash
set -e

echo "🚀 AuRAG Setup (SPHR retrieval + RDG grammar-constrained decoding)"
echo "========================================================"

# ==========================================
# 1. Python Environment Setup
# ==========================================
echo ""
echo "📦 Python Setup..."
echo "Checking Python..."
python3 --version || { echo "❌ Python 3 required"; exit 1; }

echo "Creating venv..."
python3 -m venv .venv

echo "Activating venv..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Creating data folders..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/vector_db
mkdir -p models

echo "✅ Python environment ready"

# ==========================================
# 2. Primary Mode: Download GGUF Model
# ==========================================
echo ""
echo "🤖 Primary Mode: GGUF Model Setup..."

MODEL_PATH="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model already exists: $MODEL_PATH"
else
    echo "Downloading Llama-3.1-8B-Instruct GGUF (~4.9GB)..."
    echo "This may take 10-20 minutes depending on your connection."
    echo "(Press Ctrl+C to skip and download the model manually later)"
    
    # Check if curl or wget available
    if command -v curl &> /dev/null; then
        curl -L -o "$MODEL_PATH" "$MODEL_URL" --progress-bar || {
            echo "⚠️ Download failed or cancelled. Download the model manually into models/."
            rm -f "$MODEL_PATH"
        }
    elif command -v wget &> /dev/null; then
        wget -O "$MODEL_PATH" "$MODEL_URL" --show-progress || {
            echo "⚠️ Download failed or cancelled. Download the model manually into models/."
            rm -f "$MODEL_PATH"
        }
    else
        echo "⚠️ Neither curl nor wget found. Download the model manually into models/."
    fi
    
    if [ -f "$MODEL_PATH" ]; then
        echo "✅ Model downloaded successfully"
    fi
fi

# ==========================================
# Summary
# ==========================================
echo ""
echo "========================================================"
echo "✅ AuRAG Setup Complete!"
echo "========================================================"
echo ""
echo "Architecture:"
echo "  • Layer 1: SPHR (Hierarchical Chunking & Retrieval)"
echo "  • Layer 2: RDG (Citation-Constrained Generation)"
echo ""
echo "Model:"
if [ -f "$MODEL_PATH" ]; then
    echo "  ✅ $MODEL_PATH (grammar-constrained decoding via llama-cpp-python)"
else
    echo "  ⚠️ GGUF model not found at $MODEL_PATH"
fi
echo ""
echo "Next steps:"
echo "1. Add contracts to data/raw/ folder"
echo "2. Run: source .venv/bin/activate && python src/AuRAG.py"
echo ""
