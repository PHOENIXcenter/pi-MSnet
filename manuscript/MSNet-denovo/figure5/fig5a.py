#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Title:
    Figure 5a - Performance comparison between π-HelixNovo-raw and
    π-HelixNovo-MSNet

Description:
    This script generates Figure 5a for the manuscript. It compares the peptide
    recall values of π-HelixNovo-raw and π-HelixNovo-MSNet across multiple
    datasets, calculates the relative improvement percentage of
    π-HelixNovo-MSNet over π-HelixNovo-raw, and visualizes the performance
    comparison as a line chart.

    The relative improvement is calculated as:

        (pep_recall_msnet - pep_recall_raw) / pep_recall_raw * 100

Input:
    - fig5a_summary_helixraw.csv:
        Summary table containing peptide recall values for π-HelixNovo-raw.
        Required columns:
            - dataset
            - pep_recall

    - fig5a_summary_helixmsnet.csv:
        Summary table containing peptide recall values for π-HelixNovo-MSNet.
        Required columns:
            - dataset
            - pep_recall

Output:
    - performance_comparison_with_improvement.csv:
        Merged performance comparison table with relative improvement values.

    - performance_comparison_linechart_light.svg:
        Publication-ready line chart for Figure 5a.

Author:
    Tianze Ling, Ph.D. candidate @ Tsinghua University and 
    National Center for Protein Sciences (Beijing)

Contact:
    tianzeling98@outlook.com

License:
    Copyright © 2026 [Your Name] / [Your Institution].
    All rights reserved unless otherwise specified.

Usage:
    python fig5a.py

Notes:
    - The dataset "Haloarcula marismortui" is excluded from the comparison,
      consistent with the manuscript figure preparation.
    - Arial is used as the preferred sans-serif font for publication-style
      plotting. If Arial is unavailable on the local system, matplotlib will
      fall back to the default sans-serif font.
===============================================================================
"""

import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


# =============================================================================
# Configuration
# =============================================================================

RAW_INPUT_FILE = "fig5a_summary_helixraw.csv"
MSNET_INPUT_FILE = "fig5a_summary_helixmsnet.csv"

OUTPUT_TABLE_FILE = "performance_comparison_with_improvement.csv"
OUTPUT_FIGURE_FILE = "performance_comparison_linechart_light.svg"

EXCLUDED_DATASET = "Haloarcula marismortui"
DATASET_LABEL_WRAP_WIDTH = 18

FIGURE_SIZE = (12, 6)
FIGURE_DPI = 300

RAW_MODEL_LABEL = "π-HelixNovo-raw"
MSNET_MODEL_LABEL = "π-HelixNovo-MSNet"

RAW_COLOR = "#82B0D2"      # Light blue
MSNET_COLOR = "#FA7F6F"    # Light coral


# =============================================================================
# Plot style settings
# =============================================================================

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# Data loading and processing
# =============================================================================

# Load model performance summaries.
df_raw = pd.read_csv(RAW_INPUT_FILE)
df_msnet = pd.read_csv(MSNET_INPUT_FILE)

# Merge peptide recall values from the two models by dataset.
performance_df = pd.merge(
    df_raw[["dataset", "pep_recall"]],
    df_msnet[["dataset", "pep_recall"]],
    on="dataset",
    suffixes=("_raw", "_msnet")
)

# Exclude the specified dataset from the figure.
performance_df = performance_df[
    performance_df["dataset"] != EXCLUDED_DATASET
]

# Sort datasets alphabetically and wrap long dataset names for better readability.
performance_df = performance_df.sort_values(by="dataset").reset_index(drop=True)
performance_df["dataset_wrapped"] = performance_df["dataset"].apply(
    lambda dataset_name: textwrap.fill(dataset_name, width=DATASET_LABEL_WRAP_WIDTH)
)

# Calculate the relative improvement percentage:
#     (MSNet - raw) / raw * 100
performance_df["relative_imp_pct"] = (
    (
        performance_df["pep_recall_msnet"] -
        performance_df["pep_recall_raw"]
    ) / performance_df["pep_recall_raw"]
) * 100


# =============================================================================
# Report and export processed data
# =============================================================================

print("Relative performance improvement by dataset "
      "(π-HelixNovo-MSNet vs π-HelixNovo-raw):")
print("-" * 72)

for _, row in performance_df.iterrows():
    print(f"{row['dataset']:<45} {row['relative_imp_pct']:>6.2f}%")

print("-" * 72)

# Save the merged comparison table with relative improvement values.
performance_df.to_csv(OUTPUT_TABLE_FILE, index=False)


# =============================================================================
# Plot Figure 5a
# =============================================================================

fig, ax = plt.subplots(figsize=FIGURE_SIZE)

x_positions = range(len(performance_df))
y_raw = performance_df["pep_recall_raw"] * 100
y_msnet = performance_df["pep_recall_msnet"] * 100

# Plot π-HelixNovo-raw performance.
ax.plot(
    x_positions,
    y_raw,
    marker="o",
    linestyle="-",
    color=RAW_COLOR,
    linewidth=2,
    markersize=8,
    label=RAW_MODEL_LABEL
)

# Plot π-HelixNovo-MSNet performance.
ax.plot(
    x_positions,
    y_msnet,
    marker="s",
    linestyle="-",
    color=MSNET_COLOR,
    linewidth=2,
    markersize=8,
    label=MSNET_MODEL_LABEL
)


# =============================================================================
# Add point labels
# =============================================================================

for i in x_positions:
    # Label for π-HelixNovo-raw, placed below each point.
    ax.annotate(
        f"{y_raw[i]:.1f}%",
        (i, y_raw[i]),
        textcoords="offset points",
        xytext=(0, -18),
        ha="center",
        va="top",
        fontsize=10,
        color=RAW_COLOR,
        weight="bold"
    )

    # Label for π-HelixNovo-MSNet, placed above each point.
    ax.annotate(
        f"{y_msnet[i]:.1f}%",
        (i, y_msnet[i]),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        va="bottom",
        fontsize=10,
        color=MSNET_COLOR,
        weight="bold"
    )


# =============================================================================
# Format axes and figure appearance
# =============================================================================

ax.set_xticks(x_positions)
ax.set_xticklabels(
    performance_df["dataset_wrapped"],
    rotation=45,
    ha="right",
    fontsize=12
)

ax.set_ylabel("Peptide precision", fontsize=14, weight="bold")
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

ax.legend(
    fontsize=15,
    loc="upper left",
    bbox_to_anchor=(0, 1.1)
)

# Add a light horizontal dashed grid to improve readability.
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Remove unnecessary spines for a cleaner publication-style figure.
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_FIGURE_FILE, dpi=FIGURE_DPI)
plt.show()