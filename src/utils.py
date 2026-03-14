import gc
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

def get_gpu_memory_usage() -> Optional[float]:
    """
    Returns current GPU memory allocated in MB.
    Returns None if CUDA is not available.
    Used to track memory before and after model loading
    to measure the true memory footprint of each technique.
    """
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024 ** 2) # MB

def load_model_and_tokenizer(
    model_name: Optional[str] = None,
    technique: str = "float32",
    device: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads TinyLlama with the specified optimization technique.
    Each technique requires different loading configuration —
    this function centralizes all loading logic so experiment
    scripts only need tfo call one function with a technique name.

    Supported techniques:
        float32  — baseline, no optimization
        float16  — half precision, 2x memory reduction
        int8     — LLM.int8() quantization via bitsandbytes
        int4     — NF4 quantization via bitsandbytes
        gptq     — pre-quantized GPTQ from TheBloke
        awq      — pre-quantized AWQ from TheBloke

    Args:
        model_name: HuggingFace model name. Defaults to config value.
        technique: Optimization technique to apply.
        device: Target device. Defaults to CUDA if available.
    
    Returns:
        Tuple of (model, tokenizer).
    """
    from src.benchmark_core import CONFIG, DEVICE
    
    if model_name is None:
        model_name = CONFIG['model']['name']
    
    if device is None:
        device = DEVICE
    
    cache_dir = CONFIG['model']['cache_dir']
    
    print(f"    loading {technique} model: {model_name}")
    baseline_mb = get_gpu_memory_usage()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    
    if technique == "float32":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
            cache_dir=cache_dir,
        )

    elif technique == "float16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            cache_dir=cache_dir,
        )

    elif technique == "int8":
        quant_cfg = CONFIG["quantization"]["int8"]
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=quant_cfg["load_in_8bit"],
            llm_int8_threshold=quant_cfg["llm_int8_threshold"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
            cache_dir=cache_dir,
        )

    elif technique == "int4":
        quant_cfg = CONFIG["quantization"]["int4_nf4"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg["load_in_4bit"],
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
            cache_dir=cache_dir,
        )

    elif technique == "gptq":
        gptq_cfg = CONFIG["quantization"]["gptq"]
        model = AutoModelForCausalLM.from_pretrained(
            gptq_cfg["model_name"],
            device_map=device,
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            gptq_cfg["model_name"],
            cache_dir=cache_dir,
        )

    elif technique == "awq":
        awq_cfg = CONFIG["quantization"]["awq"]
        model = AutoModelForCausalLM.from_pretrained(
            awq_cfg["model_name"],
            device_map=device,
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            awq_cfg["model_name"],
            cache_dir=cache_dir,
        )
    
    elif technique == "compiled":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device,
            cache_dir=cache_dir
        )
        model = torch.compile(model, mode="default")

    else:
        raise ValueError(
            f"Unknown technique: {technique}. "
            f"Supported: float32, float16, int8, int4, gptq, awq"
        )

    model.eval()

    after_mb = get_gpu_memory_usage()
    if baseline_mb is not None and after_mb is not None:
        footprint = after_mb - baseline_mb
        print(f"  model loaded — memory footprint: {footprint:.1f}MB")

    return model, tokenizer

def unload_model(model: AutoModelForCausalLM) -> None:
    """
    Removes model from GPU memory and clears CUDA cache.
    Must be called between experiments to prevent memory
    fragmentation from affecting subsequent measurements.
    Without explicit unloading, PyTorch holds cached memory
    even after the model variable goes out of scope.
    """
    del model
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_experiment_header(technique: str, experiment_num: int, total: int) -> None:
    """
    Prints a formatted header to terminal before each experiment runs.
    Provides visual progress tracking during long benchmark sessions.

    Args:
        technique: Name of the current technique being benchmarked.
        experiment_num: Current experiment number (1-indexed).
        total: Total number of experiments in this session.
    """
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Experiment {experiment_num}/{total}: {technique.upper()}")
    print(f"{bar}")