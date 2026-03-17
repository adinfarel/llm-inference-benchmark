import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_device():
    """
    Returns the available compute device.
    Prefers CUDA if available, falls back to CPU.
    This determines where all tensors and model weights will live
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict:
    """
    Loads the central experiment configuration from YAML.
    All hyperparameters - model name, batch sizes, sequence lenghts -
    are read from this single file to ensure consistency across experiments.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

DEVICE = get_device()
CONFIG = load_config()
 
def format_prompt(user_message: str, system_message: Optional[str] = None) -> str:
    """
    Formats a user message into TinyLlama's chat template.
    TinyLlama expects a specific structure with system and user tags.
    Without this formatting, the model output becomes unpredictable
    and metrics are not comparable across experiments.
    
    Args:
        user_message: The actual question or instruction.
        system_message: Optional system context. Defaults to config value.
    
    Returns:
        Formatted prompt string ready for tokenization
    """
    if system_message is None:
        system_message = CONFIG["prompt"]["system"]
    
    return (
        f"<|system|>\n{system_message}\n"
        f"<|user|>\n{user_message}\n"
        f"<|assistant|>\n"
    )

def measure_ttft(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    warmup: bool = True,
    is_onnx: bool = False
) -> float:
    """
    Measures Time to First Token (TTFT) in milliseconds.
    TTFT captures the prefill phase — how long the model takes
    to process the entire input prompt and produce the first output token.
    This is memory bandwidth bound: longer prompts = higher TTFT
    because more data must be loaded from HBM before compute begins.

    Args:
        model: The loaded model to benchmark.
        tokenizer: Corresponding tokenizer.
        prompt: Already formatted prompt string.
        warmup: If True, runs one warmup pass before measuring.
                GPU kernels are cached after first run,
                so warmup ensures we measure steady-state performance.
        is_onnx: To validate whether ORT or not.

    Returns:
        TTFT in milliseconds.
    """
    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(DEVICE)
    
    attention_mask = torch.ones_like(input_ids)
    
    if warmup:
        with torch.no_grad():
            if is_onnx:
                _ = model.run(
                    ['logits'], 
                    {
                        "input_ids": input_ids.cpu().numpy(),
                        "attention_mask": torch.ones_like(input_ids).cpu().numpy()
                    }
                )
            else:
                _ = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    do_sample=False
                )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t_start = time.perf_counter()
    
    with torch.no_grad():
        if is_onnx:
            _ = model.run(
                ['logits'],
                {
                    "input_ids": input_ids.cpu().numpy(),
                    "attention_mask": torch.ones_like(input_ids).cpu().numpy()
                }
            )
        else:
            _ = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False
            )
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t_end = time.perf_counter()
    
    return (t_end - t_start) * 1000 # MS

def measure_itl_and_tpot(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: Optional[int] = None,
    force_max_tokens: bool = True,
    is_onnx: bool = False
) -> Tuple[List[float], float, int]:
    """
    Measures Inter-Token Latency (ITL) for each generated token
    and computes Time Per Output Token (TPOT) as the mean ITL.

    ITL is measured per token position, not just as an average.
    This per-position granularity is what allows us to observe
    KV cache pressure — ITL should rise as the cache grows larger
    because each decode step must attend over more tokens in HBM.

    TPOT is the mean of all ITL values and represents the average
    decode speed — this is what most papers report, but it hides
    the per-position behavior that ITL captures.

    Args:
        model: The loaded model to benchmark.
        tokenizer: Corresponding tokenizer.
        prompt: Already formatted prompt string.
        max_new_tokens: Number of tokens to generate.
                        Defaults to config value.
        force_max_tokens: If True, ignore EOS and generate exactly
                          max_new_tokens. Use True for latency benchmark,
                          False for quality evaluation.
        is_onnx: To validate whether ORT or not.
    Returns:
        Tuple of (itl_list, tpot, n_generated) where:
            itl_list: ITL in milliseconds for each token position.
            tpot: Mean ITL across all positions in milliseconds.
            n_generated: Total token generated 
    """
    if max_new_tokens is None:
        max_new_tokens = CONFIG["benchmark"]["max_new_tokens"]
    
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.to(DEVICE)
    
    past_key_values = None
    generated = input_ids.clone()
    itl_list = []
    
    for step in range(max_new_tokens):
        current_input = generated if past_key_values is None else generated[:, -1:]
        current_mask = torch.ones(
            1, generated.shape[1],
            dtype=torch.long,
            device=DEVICE
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t_start = time.perf_counter()
        
        with torch.no_grad():
            if is_onnx:
                inputs = {
                    "input_ids": current_input.cpu().numpy(),
                    "attention_mask": current_mask.cpu().numpy()
                }
                logits = torch.tensor(
                    model.run(['logits'], inputs)[0]
                ).to(DEVICE)
                
                next_token = torch.argmax(
                    logits[:, -1, :], dim=-1, keepdim=True
                )
                past_key_values = None
            else:   
                outputs = model(
                    current_input,
                    attention_mask=current_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                next_token = torch.argmax(
                    outputs.logits[:, -1, :],
                    dim=-1, keepdim=True
                )
                past_key_values = outputs.past_key_values
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t_end = time.perf_counter()
        
        itl_ms = (t_end - t_start) * 1000 # MS
        itl_list.append(itl_ms)
        
        generated = torch.cat([generated, next_token], dim=-1)
        
        if not force_max_tokens:
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    tpot = float(np.mean(itl_list))
    n_generated = len(itl_list)
    
    return itl_list, tpot, n_generated

def measure_perplexity(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    max_samples: Optional[int] = None,
    stride: Optional[int] = None,
    is_onnx: bool = False
) -> float:
    """
    Computes perplexity on WikiText-103 using a sliding window approach.
    Perplexity measures how well the model predicts real text —
    lower is better. This is the quality metric that tells us whether
    an optimization technique preserved model intelligence or damaged it.

    Sliding window is used instead of naive chunking because it gives
    each token sufficient context from previous tokens, producing
    a more accurate and less biased perplexity estimate.

    Args:
        model: The loaded model to evaluate.
        tokenizer: Corresponding tokenizer.
        max_samples: Number of WikiText samples to evaluate.
                     Defaults to config value.
        stride: Sliding window stride in tokens.
                Defaults to config value.
        is_onnx: To validate whether onnxruntime or not

    Returns:
        Perplexity score as a float. Lower is better.
    """
    from datasets import load_dataset
    
    if max_samples is None:
        max_samples = CONFIG['evaluation']['max_samples']
    
    if stride is None:
        stride = CONFIG['evaluation']['stride']
    
    dataset = load_dataset(
        CONFIG['evaluation']['dataset'],
        CONFIG['evaluation']['dataset_config'],
        split='test'
    )
    
    samples = [s for s in dataset['text'] if len(s.strip()) > 0][:max_samples]
    full_text = "\n\n".join(samples)
    
    encodings = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False
    )    
    
    input_ids = encodings.input_ids.to(DEVICE)
    seq_len = input_ids.shape[1]
    max_length = CONFIG['benchmark'].get('max_length', 2048)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
    
    nlls = []
    prev_end = 0
    
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end
        
        window_input = input_ids[:, begin:end]
        
        target_ids = window_input.clone()
        target_ids[:, :-target_len] = -100
        
        with torch.no_grad():
            if is_onnx:
                inputs = {
                    "input_ids": window_input.cpu().numpy(),
                    "attention_mask": torch.ones_like(
                        window_input
                    ).cpu().numpy()
                }
                
                logits = torch.tensor(
                    model.run(['logits'], inputs)[0]
                )
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = target_ids[:, 1:].contiguous()
                
                nll = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            else:
                outputs = model(
                    window_input,
                    labels=target_ids
                )
                nll = outputs.loss * target_len
                
        nlls.append(nll)
        prev_end = end
        
        if end == seq_len:
            break
    
    total_nll = torch.stack(nlls).sum()
    perplexity = torch.exp(total_nll / prev_end).item()
    
    return perplexity

def stabilize_environment() -> None:
    """
    Clears GPU memory, Python garbage, and CUDA cache
    before each experiment to ensure a clean hardware state.
    Without this, residual memory fragments and cached kernels
    from previous experiments can affect timing measurements,
    making results inconsistent and not comparable across techniques.
    """
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def generate_measure(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: Optional[dict] = None,
    label: str = "experiment",
    measure_perplexity_score: bool = True,
    is_onnx: bool = False,
    enable_profiler: bool = False,
    custom_prompt: Optional[str] = None
) -> dict:
    """
    Runs the full benchmark pipeline for a single model configuration.
    Calls measure_ttft, measure_itl_and_tpot, and optionally
    measure_perplexity across n_runs, then aggregates results
    into p50, p90, p99 latency statistics.

    Optionally runs PyTorch profiler to capture CUDA kernel activity,
    memory bandwidth usage, and operator-level timing — this is the
    hardware-level evidence that explains why one technique is faster
    than another, not just that it is faster.

    Args:
        model: PyTorch model or OnnxRuntime InferenceSession.
        tokenizer: Corresponding tokenizer.
        config: Optional override config. Defaults to global CONFIG.
        label: Human-readable name for this experiment
               e.g. "float32", "int4_nf4", "gptq".
        measure_perplexity_score: If True, compute perplexity
                                  on WikiText-103.
        is_onnx: If True, uses ONNX path in perplexity measurement.
        enable_profiler: If True, runs PyTorch profiler and saves
                         chrome trace to results/traces/.

    Returns:
        Dictionary containing all benchmark metrics for this experiment.
    """
    if config is None:
        config = CONFIG
    
    cfg = config['benchmark']
    prompt_cfg = config['prompt']
    results_cfg = config['results']
    
    prompt = custom_prompt if custom_prompt is not None else format_prompt(prompt_cfg['user'])
    n_runs = cfg['n_runs']
    max_new_tokens = cfg['max_new_tokens']
    
    stabilize_environment()
    
    # warmup run - not measured, just to prime GPU kernel cache
    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(DEVICE)
    
    with torch.no_grad():
        _ = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=1,
            do_sample=False # Greedy search
        ) if not is_onnx else model.run(
            ['logits'],
            {
                "input_ids": input_ids.cpu().numpy(),
                "attention_mask": torch.ones_like(input_ids).cpu().numpy()
            }
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    
    # collect TTFT across n_runs
    ttft_runs = []
    
    for _ in range(n_runs):
        ttft = measure_ttft(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            warmup=False,
            is_onnx=is_onnx
        )
        ttft_runs.append(ttft)
    
    # collect ITL across n_runs
    all_itl = []
    tpot_runs = []
    n_generated_runs = []
    
    for _ in range(n_runs):
        itl_list, tpot, n_generated = measure_itl_and_tpot(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            force_max_tokens=True,
            is_onnx=is_onnx
        )    
        all_itl.extend(itl_list)
        tpot_runs.append(tpot)
        n_generated_runs.append(n_generated)
    
    # profiler - only runs once, seperate from measurement loop
    trace_path = None
    
    if enable_profiler:
        trace_dir = Path(results_cfg['traces_dir'])
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = str(trace_dir / f"{label}_trace.json")
        
        profiler_input = tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids.to(DEVICE)
        

        profiler_mask = torch.ones_like(profiler_input)
        profiler_steps = CONFIG['profiler']['max_new_tokens']
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=CONFIG['profiler']['record_shapes'],
            profile_memory=CONFIG['profiler']['profile_memory'],
            with_stack=CONFIG['profiler']['with_stack']
        ) as prof:
            with torch.no_grad():
                if is_onnx:
                    generated = profiler_input.clone()
                    
                    for _ in range(profiler_steps):
                        inputs = {
                            "input_ids": generated.cpu().numpy(),
                            "attention_mask": torch.ones(
                                1, generated.shape[1],
                                dtype=torch.long).numpy()
                        }
                        
                        logits = torch.tensor(model.run(
                            ['logits'], inputs)[0]
                        )
                        
                        next_token = torch.argmax(
                            logits[:, -1, :], dim=-1, keepdim=True
                        )
                        
                        generated = torch.cat([generated, next_token], dim=-1)
                else:
                    _ = model.generate(
                        profiler_input,
                        attention_mask=profiler_mask,
                        max_new_tokens=CONFIG['profiler']['max_new_tokens'],
                        do_sample=False
                    )
        
        prof.export_chrome_trace(trace_path)
        
    # aggregate TTFT statistics
    ttft_p50    = float(np.percentile(ttft_runs, 50))
    ttft_p90    = float(np.percentile(ttft_runs, 90))
    ttft_p99    = float(np.percentile(ttft_runs, 99))
    ttft_mean   = float(np.mean(ttft_runs))
    
    # aggregate ITL statistics across all runs
    itl_p50     = float(np.percentile(all_itl, 50))        
    itl_p90     = float(np.percentile(all_itl, 90))        
    itl_p99     = float(np.percentile(all_itl, 99))        
    itl_mean    = float(np.mean(all_itl))
    itl_std     = float(np.std(all_itl))
    
    # TPOT - mean across runs
    tpot_mean   = float(np.mean(tpot_runs))
    
    # E2E latency - TTFT + total decode time
    # total decode time = tpot_mean * mean tokens generated
    mean_n_generated = float(np.mean(n_generated_runs))
    e2e_ms      = ttft_mean + (tpot_mean * mean_n_generated)
    
    # throughput - tokens per second
    # tpot_mean is ms per token, convert to seconds
    throughput_tps   = 1000.0 / tpot_mean if tpot_mean > 0 else 0.0
    
    # perplexity - run once, not n_runs time
    perplexity = None
    if measure_perplexity_score:
        perplexity = measure_perplexity(
            model=model,
            tokenizer=tokenizer,
            is_onnx=is_onnx
        )
    
    # peak GPU memory usage in MB
    peak_memory_mb = None
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
    results = {
        "label": label,
        "ttft_p50_ms": ttft_p50,
        "ttft_p90_ms": ttft_p90,
        "ttft_p99_ms": ttft_p99,
        "ttft_mean_ms": ttft_mean,
        "itl_p50_ms": itl_p50,
        "itl_p90_ms": itl_p90,
        "itl_p99_ms": itl_p99,
        "itl_mean_ms": itl_mean,
        "itl_std_ms": itl_std,
        "itl_per_position": all_itl,
        "tpot_mean_ms": tpot_mean,
        "e2e_ms": e2e_ms,
        "throughput_tps": throughput_tps,
        "mean_n_generated": mean_n_generated,
        "peak_memory_mb": peak_memory_mb,
        "perplexity": perplexity,
        "n_runs": n_runs,
        "trace_path": trace_path,
    }
    
    return results