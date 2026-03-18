"""
Batching Benchmark — experiments/run_batching.py

Measures throughput and latency across batch sizes 1 to 32
to find the optimal batch size for sustained throughput on T4.

At batch size 1, GPU parallelism is severely underutilized —
weights are loaded from HBM to process just one request.
At larger batch sizes, the same weight load serves multiple
requests simultaneously, amortizing the IO cost.
But memory pressure grows with batch size, and beyond a point,
latency per request becomes unacceptable.

Research question answered:
    Q5: What is the optimal batch size for throughput on a T4?
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_core import CONFIG, DEVICE, format_prompt
from src.metrics import save_results, summarize_results
from src.utils import (
    load_model_and_tokenizer,
    unload_model,
    print_experiment_header
)

import numpy as np
import torch

BATCH_SIZES     = CONFIG['batching']['batch_sizes']
MAX_NEW_TOKENS  = CONFIG['batching']['max_new_tokens']
N_RUNS          = CONFIG['benchmark']['n_runs']

def measure_batch_throughput(
    model,
    tokenizer,
    batch_size: int, 
    max_new_tokens: int,
    n_runs: int,
) -> dict:
    """
    Measures throughput and per-request latency at a given batch size.
    Constructs a batch of identical prompts, runs generation,
    measures wall-clock time, computes tokens per second.

    Throughput = (batch_size × max_new_tokens) / wall_clock_seconds
    Latency per request = wall_clock_seconds / batch_size

    Args:
        model: Loaded float16 model.
        tokenizer: Corresponding tokenizer.
        batch_size: Number of requests in one batch.
        max_new_tokens: Tokens to generate per request.
        n_runs: Number of measurement runs.

    Returns:
        Dict with throughput and latency metrics.
    """
    prompt  = format_prompt(CONFIG['prompt']['user'])
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)
    
    input_ids       = encoded.input_ids.repeat(batch_size, 1)
    attention_mask  = encoded.attention_mask.repeat(batch_size, 1)
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measurement runs
    latency_runs, throughput_runs = [], []
    
    for _ in range(N_RUNS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t_start     = time.perf_counter()
        
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t_end       = time.perf_counter()
        
        wall_clock_s    = t_end - t_start
        total_tokens    = batch_size * max_new_tokens
        tps             = total_tokens / wall_clock_s
        latency_per_req = (wall_clock_s / batch_size) * 1000 # MS
        
        latency_runs.append(latency_per_req)
        throughput_runs.append(tps)
    
    return {
        "throughput_p50_tps":    float(np.percentile(throughput_runs, 50)),
        "throughput_mean_tps":   float(np.mean(throughput_runs)),
        "latency_p50_ms":        float(np.percentile(latency_runs, 50)),
        "latency_p99_ms":        float(np.percentile(latency_runs, 99)),
        "latency_mean_ms":       float(np.mean(latency_runs)),
    }

def run_batching(
    batch_sizes: list = None,
    max_new_tokens: int = None,
    save: bool = False,
) -> list:
    """
    Runs throughput benchmark across all batch sizes.
    Each batch size is measured independently with the same model
    loaded throughout — no reload between batch sizes since the
    model weights do not change, only the input batch dimension.

    Args:
        batch_sizes: List of batch sizes to benchmark.
                     Defaults to config value [1, 2, 4, 8, 16, 32].
        max_new_tokens: Tokens to generate per request per batch.
                        Defaults to config value (50).
        save: If True, persists results to CSV and JSON.

    Returns:
        List of result dicts, one per batch size.
    """
    if batch_sizes is None:
        batch_sizes = BATCH_SIZES
    
    
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    
    all_results = []
    total       = len(batch_sizes)
    
    model, tokenizer    = load_model_and_tokenizer(technique="float16")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    for idx, batch_size in enumerate(batch_sizes, start=1):
        label   = f"batch_size_{batch_size}"
        print_experiment_header(label, idx, total)
        
        try:
            metrics = measure_batch_throughput(
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                n_runs=N_RUNS,
            )
            result = {
                "label":      label,
                "batch_size": batch_size,
                **metrics,
            }
            print(f"    throughput: {result['throughput_p50_tps']:.1f} tps  "
            f"latency_p50: {result['latency_p50_ms']:.1f}ms")
        
        except RuntimeError as e:
            print(f"    OOM at batch_size={batch_size}: {e}")
            result = {
                "label":             label,
                "batch_size":        batch_size,
                "throughput_p50_tps": None,
                "throughput_mean_tps": None,
                "latency_p50_ms":    None,
                "latency_p99_ms":    None,
                "latency_mean_ms":   None,
                "error":             "OOM",
            }
        
        if save:
            save_results(result)
        
        all_results.append(result)
    
    unload_model(model)
    
    print("\n" + "=" * 60)
    print("BATCHING SUMMARY")
    print("=" * 60)
    for r in all_results:
        if r.get("error"):
            print(f"  batch={r['batch_size']:2d}: OOM")
        else:
            print(f"  batch={r['batch_size']:2d}: "
                  f"tps={r['throughput_p50_tps']:.1f}  "
                  f"latency={r['latency_p50_ms']:.1f}ms")
    print("=" * 60)

    return all_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run batching benchmark on TinyLlama 1.1B"
    )

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Batch sizes to benchmark. Default: from config. "
             "Example: --batch-sizes 1 4 8 16"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Tokens to generate per request. Default: from config (50)."
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Dry run — do not save results to disk.",
    )

    args = parser.parse_args()

    run_batching(
        batch_sizes=args.batch_sizes,
        max_new_tokens=args.max_tokens,
        save=not args.no_save,
    )