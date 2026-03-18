"""
KV Cache Pressure Benchmark — experiments/run_kv_cache.py

Measures ITL per token position across 1000 generated tokens
to identify the inflection point where KV cache size begins
to visibly increase inter-token latency.

Every decode step, the model reads the entire KV cache from HBM.
As generation grows, cache grows, HBM traffic grows, ITL rises.
This experiment maps exactly where that rise becomes visible.

Research question answered:
    Q4: At which token position does KV cache pressure become visible?
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_core import generate_measure, CONFIG, DEVICE
from src.metrics import save_results, summarize_results
from src.utils import (
    load_model_and_tokenizer,
    unload_model,
    print_experiment_header,
)

GENERATE_LENGTHS = CONFIG['kv_cache']['generate_lengths']

def run_kv_cache(
    generate_lengths: list = None,
    enable_profiler: bool = False,
    save: bool = False,
) -> list:
    """
    Benchmarks KV cache pressure across multiple generation lengths.
    For each length, loads float16 model, generates that many tokens,
    records ITL at every position, then unloads.

    The critical output is itl_per_position — a list of ITL values
    at each token position. The inflection point where ITL begins
    rising consistently is the answer to Q4.

    Runs multiple generation lengths to confirm the inflection point
    is consistent and not an artifact of a single run.

    Args:
        generate_lengths: List of token counts to generate.
                          Defaults to config value [100, 300, 500, 750, 1000].
        enable_profiler: If True, saves Chrome trace.
                         Adds overhead — use only when needed.
        save: If True, persists results to CSV and JSON.

    Returns:
        List of result dicts, one per generation length.
    """
    if generate_lengths is None:
        generate_lengths = GENERATE_LENGTHS
    
    all_results = []
    total = len(generate_lengths)
    
    for idx, max_tokens in enumerate(generate_lengths):
        label   = f"kv_cache_{max_tokens}tok"
        print_experiment_header(label, idx + 1, total)
        
        model, tokenizer = load_model_and_tokenizer(
            technique='float16'
        )
        
        kv_config = dict(CONFIG)
        kv_config['benchmark'] = dict(CONFIG['benchmark'])
        kv_config['benchmark']['max_new_tokens'] = max_tokens
        
        result = generate_measure(
            model=model,
            tokenizer=tokenizer,
            config=kv_config,
            label=label,
            enable_profiler=enable_profiler,
            measure_perplexity_score=False,
        )
        
        result['max_new_tokens'] = max_tokens
        
        if save:
            save_results(result)
        
        all_results.append(result)
        unload_model(model)
    
    summarize_results(all_results)
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run KV cache pressure benchmark on TinyLlama 1.1B"
    )
    
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=None,
        help="Generation lengths to benchmark. Default: from config. "
             "Example: --lengths 100 500 1000"
    )
    
    parser.add_argument(
        "--profiler",
        action="store_true",
        default=False,
        help="Enable PyTorch profiler and save Chrome trace.",
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Dry run — do not save results to disk.",
    )
    
    args = parser.parse_args()
    
    run_kv_cache(
        generate_lengths=args.lengths,
        enable_profiler=args.profiler,
        save=not args.no_save
    )