"""
Quantization Benchmark — experiments/run_quantization.py

Benchmarks 6 quantization techniques on TinyLlama 1.1B:
    float32  — baseline, no optimization
    float16  — half precision
    int8     — LLM.int8() via bitsandbytes
    int4     — NF4 via bitsandbytes
    gptq     — pre-quantized GPTQ from TheBloke
    awq      — pre-quantized AWQ from TheBloke

Each technique is evaluated on:
    TTFT, ITL per position, TPOT, throughput,
    peak memory usage, and perplexity on WikiText-103.

Research questions answered:
    Q1: Does int4 actually speed up on GPU?
    Q2: Which technique preserves perplexity best?
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_core import generate_measure, CONFIG
from src.metrics import save_results, summarize_results
from src.utils import (
    load_model_and_tokenizer,
    unload_model,
    print_experiment_header
)

TECHNIQUES = [
    "float32",
    "float16",
    "int8",
    "int4",
    "awq",
    "compiled"
]

def run_quantization(
    techniques: list = None,
    enable_profiler: bool = False,
    save: bool = True,
) -> list:
    """
    Runs the full quantization benchmark across all specified techniques.
    Each technique is loaded, benchmarked, and unloaded before the next
    one begins — ensuring clean GPU memory state between experiments.

    Args:
        techniques: List of techniques to benchmark.
                    Defaults to module-level TECHNIQUES list.
        enable_profiler: If True, saves Chrome trace per technique.
                         Adds overhead — use only when needed.
        save: If True, persists results to CSV and JSON.

    Returns:
        List of result dicts, one per technique.
    """
    if techniques is None:
        techniques = TECHNIQUES
    
    all_results = []
    total = len(techniques)
    
    for idx, technique in enumerate(techniques):
        print_experiment_header(technique, idx + 1, total)
        
        model, tokenizer = load_model_and_tokenizer(
            technique=technique
        )
        
        results = generate_measure(
            model=model,
            tokenizer=tokenizer,
            label=technique,
            enable_profiler=enable_profiler,
        )
        
        if save:
            save_results(results)
        
        all_results.append(results)
        
        unload_model(model)
    
    summarize_results(all_results)
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run quantization benchmark on TinyLlama 1.1B"
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        default=None,
        help="Techniques to benchmark. Default: all. "
             "Example: --techniques float32 int4 awq"
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
    
    run_quantization(
        techniques=args.techniques,
        enable_profiler=args.profiler,
        save=not args.no_save,
    )