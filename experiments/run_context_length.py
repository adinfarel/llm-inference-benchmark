"""
Context Length Benchmark — experiments/run_context_length.py

Measures TTFT across varying prompt lengths to test whether
prefill time scales quadratically with sequence length.

Prefill processes all input tokens in one forward pass.
Standard attention prefill complexity is O(N²) in sequence length —
doubling prompt length should quadruple TTFT if attention dominates.
In practice, other linear-scaling operations (MLP, embeddings) dilute
the quadratic signal, but the trend should be clearly visible.

Research question answered:
    Q6: Does prompt length affect TTFT in a way consistent
        with O(N²) complexity?
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.benchmark_core import (
    generate_measure, CONFIG, DEVICE,
    format_prompt, measure_ttft
)
from src.metrics import save_results, summarize_results
from src.utils import (
    load_model_and_tokenizer,
    unload_model,
    print_experiment_header
)

import torch
from transformers import AutoTokenizer

PROMPT_LENGTHS  = CONFIG['context_length']['prompt_length']
MODEL_NAME      = CONFIG['model']['name']
CACHE_DIR       = CONFIG['model']['cache_dir']
N_RUNS          = CONFIG['benchmark']['n_runs']

def build_prompt_of_length(
    tokenizer: AutoTokenizer,
    target_length: int,
) -> str:
    """
    Constructs a prompt that tokenizes to approximately target_length tokens.
    Repeats base sentence until token count reaches target.
    Same approach as run_flash_attention.py for consistency.

    Args:
        tokenizer: TinyLlama tokenizer.
        target_length: Desired token count.

    Returns:
        Prompt string of approximately target_length tokens.
    """
    base    = CONFIG['prompt']['user']
    prompt  = base
    
    while True:
        tokens  = tokenizer(prompt, return_tensors="pt").input_ids
        if tokens.shape[1] >= target_length:
            break
        prompt += " " + base
    
    return prompt

def run_context_length(
    prompt_lengths: list = None,
    save: bool = False,
) -> list:
    """
    Runs TTFT measurement across all prompt lengths.
    Loads model once, sweeps all lengths, unloads.
    TTFT only — no token generation needed for this experiment.

    Args:
        prompt_lengths: List of token counts to benchmark.
                        Defaults to config value.
        save: If True, persists results to CSV and JSON.

    Returns:
        List of result dicts, one per prompt length.
    """
    if prompt_lengths is None:
        prompt_lengths = PROMPT_LENGTHS
    
    all_results = []
    total       = len(prompt_lenghts)
    
    
    
    model, tokenizer = load_model_and_tokenizer(
        technique="float16",
    )
    
    for idx, length in enumerate(prompt_lenghts, start=1):
        label   = f"context_len({length})"
        print_experiment_header(label, idx, total)
        
        prompt  = build_prompt_of_length(tokenizer=tokenizer, target_length=length)
        actual_len = tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids.shape[1]
        print(f"    actual length: {actual_len} tokens")
        
        # Measures ttft across n_runs
        import numpy as np
        ttft_runs   = []
        
        for _ in range(N_RUNS):
            ttft    = measure_ttft(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                warmup=False
            )
            ttft_runs.append(ttft)
        
        result  = {
            "label":          label,
            "prompt_length":  actual_len,
            "ttft_p50_ms":    float(np.percentile(ttft_runs, 50)),
            "ttft_p90_ms":    float(np.percentile(ttft_runs, 90)),
            "ttft_p99_ms":    float(np.percentile(ttft_runs, 99)),
            "ttft_mean_ms":   float(np.mean(ttft_runs)),
            "n_runs":         N_RUNS,
        }
        
        if save:
            save_results(result)
        
        all_results.append(result)
        print(f"    ttft_p50:{result['ttft_p50_ms']:.1f}ms")
    
    unload_model(model)
    
    print("\n" + "=" * 60)
    print("CONTEXT LENGTH SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"  len={r['prompt_length']:4d}: "
              f"ttft_p50={r['ttft_p50_ms']:.1f}ms")
    print("=" * 60)
    
    return all_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run context length benchmark on TinyLlama 1.1B"
    )

    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=None,
        help="Prompt lengths to benchmark. Default: from config. "
             "Example: --lengths 32 128 512 1024"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Dry run — do not save results to disk.",
    )

    args = parser.parse_args()

    run_context_length(
        prompt_lengths=args.lengths,
        save=not args.no_save,
    )