"""
Flash Attention Benchmark — experiments/run_flash_attention.py

Benchmarks standard attention vs Flash Attention 2 on TinyLlama 1.1B
across varying prompt lengths to measure ITL flattening behavior.

Each configuration is loaded fresh to ensure clean GPU state.
Prompt lengths sweep from 32 to 1024 tokens to capture the
crossover point where Flash Attention benefit becomes visible.

Research question answered:
    Q3: Does Flash Attention actually flatten ITL at long sequences?
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_core import generate_measure, CONFIG, format_prompt, DEVICE
from src.metrics import save_results, summarize_results
from src.utils import unload_model, print_experiment_header

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_LENGTHS  = CONFIG['flash_attention']['sequence_lengths']
MODEL_NAME      = CONFIG['model']['name']
CACHE_DIR       = CONFIG['model']['cache_dir']
PROMPT          = CONFIG['prompt']['user']

def load_model_with_attn(
    attn_implementation: str,
) -> tuple:
    """
    Loads TinyLlama with the specified attention implementation.
    Uses float16 as dtype baseline to isolate attention effect.

    Args:
        attn_implementation: Either "eager" for standard attention
                             or "flash_attention_2" for Flash Attention.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer   = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR
    )
    
    model       = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        attn_implementation=attn_implementation,
        device_map=DEVICE,
        cache_dir=CACHE_DIR,
    )
    
    model.eval()
    return model, tokenizer

def build_prompt_of_length(
    tokenizer: AutoTokenizer,
    target_length: int,
) -> str:
    """
    Constructs a prompt that tokenizes to approximately target_length tokens.
    Repeats a base sentence until the token count reaches the target.
    This ensures sequence length is controlled and reproducible
    across both attention implementations.

    Args:
        tokenizer: TinyLlama tokenizer.
        target_length: Desired token count for the prompt.

    Returns:
        Prompt string of approximately target_length tokens.
    """
    base    = PROMPT
    prompt  = base
    
    while True:
        tokens = tokenizer(prompt, return_tensors='pt').input_ids
        if tokens.shape[1] >= target_length:
            break
        prompt += base
    
    return prompt

def run_flash_attention(
    prompt_lengths: list = None,
    enable_profiler: bool = False,
    save: bool = False,
) -> list:
    """
    Runs Flash Attention benchmark across all prompt lengths.
    For each prompt length, loads standard attention model,
    benchmarks, unloads, then loads Flash Attention model,
    benchmarks, unloads — ensuring clean GPU state between runs.

    Args:
        prompt_lengths: List of token counts to benchmark.
                        Defaults to config value.
        enable_profiler: If True, saves Chrome trace per run.
                         Adds overhead — use only when needed.
        save: If True, persists results to CSV and JSON.

    Returns:
        List of result dicts, one per (attention_type, prompt_length) pair.
    """
    if prompt_lengths is None:
        prompt_lengths = PROMPT_LENGTHS
    
    all_results = []
    configurations = [
        ("standard", "eager"),
        ("flash_attn", "flash_attention_2"),
    ]
    total = len(prompt_lengths) * len(configurations)
    counter = 0
    
    for attn_label, attn_impl in configurations:
        for length in prompt_lengths:
            counter += 1
            label = f"{attn_label}_len({length})"
            print_experiment_header(label, counter, total)
            
            model, tokenizer = load_model_with_attn(attn_impl)
            
            # Build prompt
            prompt = build_prompt_of_length(tokenizer, length)
            actual_len = tokenizer(
                prompt, return_tensors='pt'
            ).input_ids.shape[1]
            print(f"    prompt length: {actual_len} tokens")
            
            result = generate_measure(
                model=model,
                tokenizer=tokenizer,
                label=label,
                enable_profiler=enable_profiler,
                measure_perplexity_score=False,
            )
            
            result['prompt_length'] = actual_len
            result['attn_type']     = attn_label
            
            if save:
                save_results(result)
            
            all_results.append(result)
            unload_model(model)
        
    summarize_results(all_results)
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Flash Attention benchmark on TinyLlama 1.1B"
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
        "--profiler",
        action="store_true",
        default=False,
        help="Enable PyTorch profiler and save Chrome traces.",
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Dry run - do not save results to disk."
    )
    
    args = parser.parse_args()
    
    run_flash_attention(
        prompt_lengths=args.lengths,
        enable_profiler=args.profiler,
        save=not args.no_save,
    )