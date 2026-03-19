#!/bin/bash
# Setup script for LLM Inference Benchmark
# Installs all dependencies, validates CUDA, and confirms model access.

set -e  # Exit immediately if any command fails

echo "============================================================"
echo "  LLM Inference Benchmark — Environment Setup"
echo "============================================================"

# Python version check
echo ""
echo "[1/5] Checking Python version..."
python --version
if ! python -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo "ERROR: Python 3.10+ required"
    exit 1
fi
echo "  OK"

# Install core dependencies
echo ""
echo "[2/5] Installing dependencies..."
pip install -r requirements.txt
echo "  OK"

# CUDA validation
echo ""
echo "[3/5] Validating CUDA availability..."
python -c "
import torch
if not torch.cuda.is_available():
    print('WARNING: CUDA not available — experiments will run on CPU')
    print('         Results will not match GPU benchmark numbers')
else:
    device = torch.cuda.get_device_name(0)
    vram   = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU: {device}')
    print(f'  VRAM: {vram:.1f}GB')
    print(f'  CUDA: {torch.version.cuda}')
    if vram < 14:
        print('WARNING: Less than 14GB VRAM detected')
        print('         Some experiments may OOM at large batch sizes')
    else:
        print('  OK')
"

# Results directory setup
echo ""
echo "[4/5] Creating results directories..."
mkdir -p results/metrics
mkdir -p results/figures/quantization
mkdir -p results/figures/pruning
mkdir -p results/figures/distillation
mkdir -p results/figures/flash_attention
mkdir -p results/figures/kv_cache
mkdir -p results/figures/context_length
mkdir -p results/figures/batching
mkdir -p results/figures/onnx
mkdir -p results/traces
mkdir -p model_cache
echo "  OK"

# Model access validation
echo ""
echo "[5/5] Validating model access..."
python -c "
from transformers import AutoTokenizer
print('  Downloading tokenizer for TinyLlama/TinyLlama-1.1B-Chat-v1.0...')
tokenizer = AutoTokenizer.from_pretrained(
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    cache_dir='./model_cache'
)
test = tokenizer('Hello world', return_tensors='pt')
print(f'  Tokenizer OK — vocab size: {tokenizer.vocab_size}')
print('  Model weights will be downloaded on first experiment run')
"

echo ""
echo "============================================================"
echo "  Setup complete. Run experiments with:"
echo "  python experiments/run_all.py"
echo "============================================================"