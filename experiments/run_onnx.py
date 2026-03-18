"""
ONNX Runtime Benchmark — experiments/run_onnx.py

Benchmarks TinyLlama exported to ONNX format against PyTorch
eager mode to measure dispatch overhead elimination from
static graph execution.

PyTorch eager mode dispatches ~220 kernel launches per token,
each with fixed CPU overhead before GPU execution begins.
ONNX Runtime resolves the full execution plan at load time —
inference runs with zero Python dispatch per token.

The difference in ITL between PyTorch and ONNX is the
empirical measure of dispatch overhead in PyTorch eager mode.

Research question answered:
    Q9: How much does static graph execution reduce dispatch
        overhead compared to PyTorch dynamic graph?
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_core import generate_measure, CONFIG, DEVICE
from src.metrics import save_results, summarize_results
from src.utils import (
    load_model_and_tokenizer,
    unload_model,
    print_experiment_header
)

import torch
from transformers import AutoTokenizer

ONNX_PATH   = CONFIG.get('onnx', {}).get(
    'export_path', './model_cache/tinyllama_onnx'
)
MODEL_NAME  = CONFIG['model']['name']
CACHE_DIR   = CONFIG['model']['cache_dir']


def export_to_onnx(
    model,
    tokenizer,
    export_path: str,
) -> str:
    """
    Exports TinyLlama to ONNX format using Optimum library.
    Export happens once and is cached — subsequent runs load
    from disk directly without re-exporting.

    Args:
        model: Loaded PyTorch model.
        tokenizer: Corresponding tokenizer.
        export_path: Directory to save ONNX model files.

    Returns:
        Path to exported ONNX model directory.
    """
    from onnx.onnxruntime import ORTModelForCausalLM
    
    export_dir = Path(export_path)
    
    if export_dir.exists():
        print(f"    ONNX model already exists at {export_path} — skipping export")
        return export_path

    ort_model = ORTModelForCausalLM.from_pretrained(
        MODEL_NAME,
        export=True,
        cache_dir=CACHE_DIR,
    )
    ort_model.save_pretrained(export_path)
    tokenizer.save_pretrained(export_path)

    print(f"    Export complete.")
    return export_path

def load_onnx_session(export_path: str):
    """
    Loads exported ONNX model as OnnxRuntime InferenceSession.
    Uses CUDAExecutionProvider if available, falls back to CPU.

    Args:
        export_path: Path to exported ONNX model directory.

    Returns:
        ORTModelForCausalLM session ready for inference.
    """
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    print(f"    Loading ONNX session from {export_path}...")

    ort_model = ORTModelForCausalLM.from_pretrained(
        export_path,
        provider="CUDAExecutionProvider" if torch.cuda.is_available()
                 else "CPUExecutionProvider",
    )

    tokenizer = AutoTokenizer.from_pretrained(export_path)

    return ort_model, tokenizer

def run_onnx(
    enable_profiler: bool = False,
    save: bool = False,
) -> list:
    """
    Runs ONNX vs PyTorch comparison benchmark.
    Benchmarks float16 PyTorch baseline first,
    then exports to ONNX and benchmarks ONNX session.

    Args:
        enable_profiler: If True, saves Chrome trace for both runs.
        save: If True, persists results to CSV and JSON.

    Returns:
        List of two result dicts: pytorch_float16 and onnx.
    """
    all_results = []

    # Run 1 — PyTorch float16 baseline
    print_experiment_header("pytorch_float16", 1, 2)

    model, tokenizer = load_model_and_tokenizer(technique="float16")

    pytorch_result = generate_measure(
        model=model,
        tokenizer=tokenizer,
        label="pytorch_float16",
        enable_profiler=enable_profiler,
        measure_perplexity_score=True,
    )

    if save:
        save_results(pytorch_result)

    all_results.append(pytorch_result)

    # Export to ONNX before unloading PyTorch model
    export_path = export_to_onnx(model, tokenizer, ONNX_PATH)
    unload_model(model)

    # Run 2 — ONNX Runtime
    print_experiment_header("onnx_runtime", 2, 2)

    ort_model, ort_tokenizer = load_onnx_session(export_path)

    onnx_result = generate_measure(
        model=ort_model,
        tokenizer=ort_tokenizer,
        label="onnx_runtime",
        enable_profiler=enable_profiler,
        measure_perplexity_score=True,
        is_onnx=True,
    )

    if save:
        save_results(onnx_result)

    all_results.append(onnx_result)

    # ONNX session does not need unload_model — not a PyTorch model

    summarize_results(all_results)
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ONNX vs PyTorch benchmark on TinyLlama 1.1B"
    )

    parser.add_argument(
        "--profiler",
        action="store_true",
        default=False,
        help="Enable PyTorch profiler and save Chrome traces.",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Dry run — do not save results to disk.",
    )

    args = parser.parse_args()

    run_onnx(
        enable_profiler=args.profiler,
        save=not args.no_save,
    )