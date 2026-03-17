"""
Distillation Benchmark — experiments/run_distillation.py.

Compares a larger Teacher model against its officially distilled Student model.
This measures the practical production trade-offs: TPS vs VRAM vs Perplexity.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import load_model_and_tokenizer, unload_model, print_experiment_header
from src.benchmark_core import generate_measure, CONFIG
from src.metrics import save_results, summarize_results

def run_distillation(
    enable_profiler: bool = False,
    save: bool = False
):
    """
    Benchmarks Teacher vs Student models to evaluate distillation efficiency.
    """
    TEACHER_MODEL = "openlm-research/open_llama_3b_v2"
    STUDENT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    prompt = CONFIG["prompt"]["system"] + "\n" + CONFIG["prompt"]["user"]
    
    models = [
        {"label": "teacher_3B", "name": TEACHER_MODEL},
        {"label": "student_1B", "name": STUDENT_MODEL}
    ]
    
    all_results = []
    
    for i, model_info in enumerate(models, 1):
        label       = model_info["label"]
        model_name  = model_info["name"]
        
        print_experiment_header(f"DISTILLATION: {label.upper()}", i, len(models))
        
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            technique="float16"
        )
        
        print(f"\nRunning benchmark for {label}...")
        results = generate_measure(
            model=model,
            tokenizer=tokenizer,
            label=label,
            enable_profiler=enable_profiler
        )
        
        if save:
            save_results(results=results)
        
        all_results.append(results)
        unload_model(model)
    
    summarize_results(all_results)
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run distillation benchmark, Teacher Model: openlm-research/open_llama_3b_v2"
                    "Student Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
    
    run_distillation(
        enable_profiler=args.profiler,
        save=not args.no_save,
    )