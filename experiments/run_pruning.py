"""
Pruning Benchmark — experiments/run_pruning.py

Benchmarks unstructured L1 pruning on TinyLlama 1.1B at varying sparsity levels.
Baseline is float16 to isolate the pruning effect.
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

SPARSE_LEVEL = CONFIG['pruning']['sparsity_levels']
TARGET_MODULES = CONFIG['pruning']['target_modules']

def run_pruning(
    sparse_level: list = None,
    target_modules: list = None,
    enable_profiler: bool = False,
    save: bool = False,
):
    """
    Runs the full pruning across sparse level to see impact pruning at inference
    Each pruning sparse level use baseline model float32, benchmarked, unloaded model
    To ensure clean memory of GPU
    
    Args:
        sparse_level: List of sparse that we want see how many affect pruning at inference
                      if List of sparse level None, then fallback to SPARSE_LEVEL
        target_modules: Where we did pruning at modules, 
                        Fallback response: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        enable_profiler: If True, saves Chrome trace per technique.
                         Adds overhead — use only when needed.
        save: If True, persists results to CSV and JSON.
    
    Returns:
        List of results dict, one of sparse level
    """
    if sparse_level is None:
        sparse_level = SPARSE_LEVEL
    
    if target_modules is None:
        target_modules = TARGET_MODULES
        
    sparse_level = [float(s) for s in sparse_level]
    
    all_results = []
    total = len(sparse_level)
    
    for idx, sparsity in enumerate(sparse_level):
        
        label = f"pruned_{int(sparsity*100)}pct"
        print_experiment_header(label, idx + 1, total)
        
        # Initialize model
        model, tokenizer = load_model_and_tokenizer(
            technique='float16', # We choose float16 as a baseline that we do prune
            sparsity_level=sparsity,
            target_modules=target_modules,
        )
        
        result = generate_measure(
            model=model,
            tokenizer=tokenizer,
            label=label,
            enable_profiler=enable_profiler,
        )
        
        if save:
            save_results(result)
        
        all_results.append(result)
        unload_model(model)
    
    summarize_results(all_results)
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run pruning benchmark on TinyLlama 1.1B"
    )
    
    parser.add_argument(
        "--sparsity",
        nargs="+",
        default=None,
        help="Sparse to benchmark, default [0.1, 0.3, 0,5, 0.7]"
             "Example: --sparsity 0.1 0.3, and so on..."
    )
    
    parser.add_argument(
        "--modules",
        nargs="+",
        default=None,
        help="Modules that we want affect pruning effect, default attention weight"
             "Example: --modules q_proj k_proj, and so on..."
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
        help="Dry run - do not save on disk.",
    )
    
    args = parser.parse_args()
    
    run_pruning(
        sparse_level=args.sparsity,
        target_modules=args.modules,
        enable_profiler=args.profiler,
        save=not args.no_save,
    )