"""
Run All Experiments — experiments/run_all.py

Executes all benchmark experiments in sequence.
Each experiment loads its model, runs benchmarks,
saves results, and unloads before the next begins.

Run order is designed to minimize total session time:
    1. Quantization    — multiple model loads, longest overall
    2. Pruning         — float16 base + sparsity levels
    3. Distillation    — teacher + student
    4. Flash Attention — standard + sdpa × sequence lengths
    5. KV Cache        — float16, long generation
    6. Context Length  — float16, TTFT sweep
    7. Batching        — float16, batch size sweep
    8. ONNX            — float16 + ONNX export + session
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_quantization    import run_quantization
from experiments.run_pruning         import run_pruning
from experiments.run_distillation    import run_distillation
from experiments.run_flash_attention import run_flash_attention
from experiments.run_kv_cache        import run_kv_cache
from experiments.run_context_length  import run_context_length
from experiments.run_batching        import run_batching
from experiments.run_onnx            import run_onnx

import gc
import torch

def run_all(save: bool = True) -> None:
    """
    Runs all experiments sequentially with cleanup between each.

    Args:
        save: If True, persists all results to disk.
              Default True — run_all is intended for full benchmark runs.
    """

def cleanup():
    """
    Clears GPU memory and Python garbage between experiments.
    Called between each experiment to prevent memory fragmentation
    from affecting subsequent measurements.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def run_all(save: bool = True) -> None:
    """
    Runs all experiments sequentially with cleanup between each.

    Args:
        save: If True, persists all results to disk.
              Default True — run_all is intended for full benchmark runs.
    """
    print("\n" + "=" * 70)
    print("  LLM INFERENCE BENCHMARK — FULL RUN")
    print("=" * 70)

    experiments = [
        ("Quantization",    lambda: run_quantization(save=save)),
        ("Pruning",         lambda: run_pruning(save=save)),
        ("Distillation",    lambda: run_distillation(save=save)),
        ("Flash Attention", lambda: run_flash_attention(save=save)),
        ("KV Cache",        lambda: run_kv_cache(save=save)),
        ("Context Length",  lambda: run_context_length(save=save)),
        ("Batching",        lambda: run_batching(save=save)),
        ("ONNX",            lambda: run_onnx(enable_profiler=True, save=save)),
    ]

    total   = len(experiments)
    results = {}

    for idx, (name, fn) in enumerate(experiments, 1):
        print(f"\n{'=' * 70}")
        print(f"  [{idx}/{total}] {name}")
        print(f"{'=' * 70}")

        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            results[name] = None

        cleanup()

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    for name, result in results.items():
        status = "✓" if result is not None else "✗ FAILED"
        print(f"  {status}  {name}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all LLM inference benchmark experiments"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Dry run — do not save results to disk.",
    )

    args = parser.parse_args()

    run_all(save=not args.no_save)