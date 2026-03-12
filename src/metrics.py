import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

METRICS_COLUMNS = [
    "label",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "ttft_mean_ms",
    "itl_p50_ms",
    "itl_p90_ms",
    "itl_p99_ms",
    "itl_mean_ms",
    "itl_std_ms",
    "tpot_mean_ms",
    "e2e_ms",
    "throughput_tps",
    "mean_n_generated",
    "peak_memory_mb",
    "perplexity",
    "n_runs",
    "timestamp",
]

def save_results(
    results: dict,
    metrics_dir: str = "results/metrics",
    csv_filename: str = "all_results.csv"
) -> None:
    """
    Persists benchmark results to both CSV and JSON formats.
    CSV stores scalar metrics for cross-experiment comparison.
    JSON stores the full result including itl_per_position,
    which cannot be stored in flat tabular format.

    Args:
        results: Dictionary returned by generate_and_measure().
        metrics_dir: Directory to save output files.
        csv_filename: Name of the shared CSV file across experiments.
    """
    metrics_path = Path(metrics_dir)
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    json_data = dict(results)
    json_data['timestamp'] = timestamp
    
    if isinstance(json_data.get("itl_per_position"), np.ndarray):
        json_data['itl_per_position'] = json_data['itl_per_position'].tolist()
    
    json_path = metrics_path / f"{results['label']}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    csv_row = {col: results.get(col, None) for col in METRICS_COLUMNS}
    csv_row['timestamp'] = timestamp
    
    csv_path = metrics_path / csv_filename
    file_exists = os.path.exists(csv_path)
    
    csv_df = pd.DataFrame([csv_row])
    
    if file_exists:
       csv_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        csv_df.to_csv(csv_path, mode='w', header=True, index=False)
    
    print(f"  saved → {json_path.name}")
    print(f"  saved → {csv_path.name} ({'appended' if file_exists else 'created'})")

def load_results(
    metrics_dir: str = "results/metrics",
    csv_filename: str = "all_results.csv",
) -> Optional[pd.DataFrame]:
    """
    Loads all benchmark results from CSV into a pandas DataFrame.
    Returns None if no results file exists yet.
    Used by analysis notebooks and experiment scripts to read
    previously saved results.

    Args:
        metrics_dir: Directory where results are stored.
        csv_filename: Name of the shared CSV file.

    Returns:
        DataFrame with all experiment results, or None if not found.
    """
    csv_path = Path(metrics_dir) / csv_filename
    
    if not csv_path.exists():
        print(f"    no results found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    return df

def summarize_results(
    results: List[dict],
    sort_by: str = "throughput_tps",
) -> None:
    """
    Prints a formatted summary table of benchmark results to terminal.
    Used as a quick sanity check after experiments complete —
    before opening analysis notebooks.
    Results are sorted by the specified metric, descending.

    Args:
        results: List of result dicts from generate_and_measure().
        sort_by: Metric to sort by. Default is throughput_tps.
    """
    if not results:
        print(f"no results to summarize")
        return
    
    display_cols = [
        "label",
        "ttft_p50_ms",
        "itl_p50_ms",
        "itl_p99_ms",
        "tpot_mean_ms",
        "throughput_tps",
        "peak_memory_mb",
        "perplexity",
    ]
    
    rows = []
    for r in results:
        row = {col: r.get(col, None) for col in display_cols}
        rows.append(row)
    
    df = pd.DataFrame(rows, columns=display_cols)
    
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)
    
    float_cols = {
        "ttft_p50_ms": 1,
        "itl_p50_ms": 2,
        "itl_p99_ms": 2,
        "tpot_mean_ms": 2,
        "throughput_tps": 1,
        "peak_memory_mb": 1,
        "perplexity": 4,
    }
    
    for col, decimals in float_cols.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: round(x, decimals) if pd.notna(x) else None
            )
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")
