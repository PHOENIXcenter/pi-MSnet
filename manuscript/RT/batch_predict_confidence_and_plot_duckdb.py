# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:09:32 2025

@author: FineLiu

Function: Batch add confidence to RT predictions and plot threshold-count curves (SVG)
Dependency: duckdb_calculate_confidence.py must be in the same directory
Usage: python batch_predict_confidence_and_plot_duckdb.py test.csv
Output: test_confidence.csv + confidence_*.svg (5 files)
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from duckdb_calculate_confidence import ShardedPeptideConfidenceCalculator

# ==========  Parameters (modify as needed) ==========
SUMMARY_CSV = Path('peptide_rt_statistics_summary.csv')   # produced by collect script
THRESHOLDS  = np.arange(0, 1.01, 0.05)                   # curve points
# ===================================================


def compute_confidence_for_csv(in_csv: Path) -> pd.DataFrame:
    """
    Read CSV, compute confidence for each row using the calculator,
    and return DataFrame with 5 extra confidence columns.
    """
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_CSV}")

    calculator = ShardedPeptideConfidenceCalculator(summary_df_path=str(SUMMARY_CSV))

    df = pd.read_csv(in_csv)
    required = {'Peptide', 'Tool', 'Predict'}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required}")

    # Drop rows with missing required values to avoid calculation errors
    initial_len = len(df)
    df = df.dropna(subset=['Peptide', 'Predict'])
    if len(df) < initial_len:
        print(f"[warning] Dropped {initial_len - len(df)} rows with NaN in Peptide or Predict.")

    total = len(df)
    if total == 0:
        raise ValueError("No valid rows remaining after dropping NaNs.")

    # Vectorized confidence computation using apply
    print(f"[info] Computing confidence for {total} peptides...")
    conf_cols = ['method1_quantile', 'method2_robust_zscore',
                 'method3_distance', 'method4_kde', 'average_confidence']

    def get_confidences(row):
        res = calculator.calculate_all_confidences(
            peptide_name=row['Peptide'],
            predicted_rt=row['Predict']
        )
        return pd.Series(res)

    # Show progress during apply (tqdm could be added but kept simple)
    conf_df = df.apply(get_confidences, axis=1)
    conf_df.columns = conf_cols
    out_df = pd.concat([df.reset_index(drop=True), conf_df], axis=1)
    return out_df


def plot_threshold_curves(df: pd.DataFrame, out_prefix: str) -> None:
    """
    Generate 5 SVG line plots (one per confidence method) showing
    peptide counts above varying confidence thresholds, grouped by tool.
    """
    methods = {
        'method1_quantile': 'Quantile',
        'method2_robust_zscore': 'Robust-ZScore',
        'method3_distance': 'Distance',
        'method4_kde': 'KDE',
        'average_confidence': 'Average'
    }

    tools = df['Tool'].unique()
    # Use a colormap large enough to handle many tools without cycling too soon
    cmap = plt.cm.get_cmap('tab10' if len(tools) <= 10 else 'tab20')
    palette = [cmap(i) for i in np.linspace(0, 1, len(tools))]

    for col, display_name in methods.items():
        plt.figure(figsize=(6, 4))
        for tool, color in zip(tools, palette):
            sub = df[df['Tool'] == tool]
            # Count peptides with confidence >= each threshold
            counts = [(sub[col] >= thr).sum() for thr in THRESHOLDS]
            plt.plot(THRESHOLDS, counts, label=tool, color=color, linewidth=1.8)

        plt.title(f'Peptide counts vs. confidence threshold ({display_name})')
        plt.xlabel('Confidence threshold')
        plt.ylabel('Number of peptides ≥ threshold')
        plt.legend(title='Tool', frameon=False)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        svg_name = f'{out_prefix}_{col}.svg'
        plt.savefig(svg_name, format='svg')
        plt.close()
        print(f'[save]  {svg_name}')


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python batch_predict_confidence_and_plot_duckdb.py <input.csv>')
        sys.exit(1)

    in_file = Path(sys.argv[1])
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    # Output CSV: replace extension with _confidence.csv
    out_file = in_file.with_name(f"{in_file.stem}_confidence.csv")

    print('=' * 60)
    print('Step 1/2  Computing confidence …')
    out_df = compute_confidence_for_csv(in_file)
    out_df.to_csv(out_file, index=False)
    print(f'[save]  {out_file}')

    print('=' * 60)
    print('Step 2/2  Plotting threshold-count curves (SVG) …')
    plot_threshold_curves(out_df, in_file.stem)

    print('=' * 60)
    print('All done!')


if __name__ == '__main__':
    main()