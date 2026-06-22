#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Title:
    Figure 5b - Multi-metric comparison between π-MSNet and the nine-species
    dataset without ricebean

Description:
    This script generates Figure 5b for the manuscript. It compares π-MSNet
    against the nine-species dataset without ricebean across multiple metrics,
    including data scale, sequence coverage, PTM information richness, and
    species coverage breadth.

    The figure is organized into three horizontal panels:
        1. Data scale and sequence coverage:
           PSMs, precursors, and sequences.
        2. PTM information richness:
           PTM sites and modified sequences.
        3. Species coverage breadth:
           Species.

    For each metric, the percentage increase of π-MSNet relative to the
    nine-species dataset without ricebean is calculated as:

        (π-MSNet - nine-species) / nine-species * 100

Input:
    - No external input file is required.
    - Data values are manually defined in this script.

Output:
    - msnet_diversity_three_panels.svg:
        Publication-ready three-panel bar chart for Figure 5b.

Author:
    Tianze Ling, Ph.D. candidate @ Tsinghua University and 
    National Center for Protein Sciences (Beijing)

Contact:
    tianzeling98@outlook.com

Usage:
    python fig5b.py

Notes:
    - The first value in each metric list corresponds to π-MSNet.
    - The second value in each metric list corresponds to the nine-species
      dataset without ricebean.
    - Percentage annotations indicate the relative increase of π-MSNet compared
      with the nine-species dataset without ricebean.
===============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_FIGURE_FILE = "msnet_diversity_three_panels.svg"
FIGURE_SIZE = (16, 5.8)
FIGURE_DPI = 600

MSNET_LABEL = "π-MSNet"
NINE_SPECIES_LABEL = "Nine-species w/o ricebean"

MSNET_COLOR = "#8DC9C0"
NINE_SPECIES_COLOR = "#E8C89A"
IMPROVEMENT_COLOR = "#4F81BD"

BAR_EDGE_COLOR = "white"
BAR_EDGE_WIDTH = 0.8

GROUPED_BAR_WIDTH = 0.28
SINGLE_PANEL_BAR_WIDTH = 0.15
SINGLE_PANEL_BAR_OFFSET = 0.14

TOP_MARGIN_RATIO = 1.10


# =============================================================================
# Input data
# =============================================================================

# The first value in each list corresponds to π-MSNet.
# The second value in each list corresponds to the nine-species dataset
# without ricebean.
DATA_GROUP_1 = {
    "PSMs": [1967882, 1488508],
    "Precursors": [1031514, 363770],
    "Sequences": [876858, 290468],
}

DATA_GROUP_2 = {
    "PTM sites": [595849, 417254],
    "Modified sequences": [219455, 88726],
}

DATA_GROUP_3 = {
    "Species": [31, 8],
}


# =============================================================================
# Global plotting style
# =============================================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# Helper functions
# =============================================================================

def apply_common_axis_style(axis):
    """
    Apply common axis styling to a subplot.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis object to be formatted.

    Returns
    -------
    None
        The function modifies the provided axis object in place.
    """

    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)

    for spine in axis.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#666666")


def set_scientific_yaxis(axis):
    """
    Format the y-axis using scientific notation.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis object whose y-axis should be formatted.

    Returns
    -------
    None
        The function modifies the provided axis object in place.
    """

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    axis.yaxis.set_major_formatter(formatter)


def annotate_bar_values(axis, bars, values, offset_ratio=0.012, fontsize=11):
    """
    Annotate bars with their absolute values.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis object containing the bars.
    bars : matplotlib.container.BarContainer
        Bar container returned by matplotlib.axes.Axes.bar.
    values : list of int or float
        Values to be displayed above the bars.
    offset_ratio : float, optional
        Offset above the bar as a fraction of the maximum value.
    fontsize : int, optional
        Font size of the annotation text.

    Returns
    -------
    None
        The function adds text annotations to the axis.
    """

    max_value = max(values) if max(values) > 0 else 1
    offset = (
        max_value * offset_ratio
        if max_value > 100
        else max_value * 0.04 + 0.15
    )

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=fontsize
        )


def annotate_relative_increase(
    axis,
    baseline_bars,
    msnet_values,
    baseline_values,
    offset_ratio=0.05,
    fontsize=11
):
    """
    Annotate the relative increase of π-MSNet above the baseline bars.

    The relative increase is calculated as:

        (π-MSNet - baseline) / baseline * 100

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis object containing the bars.
    baseline_bars : matplotlib.container.BarContainer
        Bars corresponding to the baseline dataset.
    msnet_values : list of int or float
        Values corresponding to π-MSNet.
    baseline_values : list of int or float
        Values corresponding to the baseline dataset.
    offset_ratio : float, optional
        Offset above the baseline bar as a fraction of the overall maximum value.
    fontsize : int, optional
        Font size of the annotation text.

    Returns
    -------
    None
        The function adds percentage annotations to the axis.
    """

    combined_values = msnet_values + baseline_values
    overall_max = max(combined_values) if max(combined_values) > 0 else 1

    for bar, msnet_value, baseline_value in zip(
        baseline_bars,
        msnet_values,
        baseline_values
    ):
        relative_increase = (
            (msnet_value - baseline_value) / baseline_value * 100
            if baseline_value != 0
            else 0
        )

        y_position = (
            bar.get_height() + overall_max * offset_ratio
            if overall_max > 100
            else bar.get_height() + overall_max * 0.12 + 0.2
        )

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            y_position,
            f"+{relative_increase:.1f}%",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=IMPROVEMENT_COLOR,
            fontweight="bold"
        )


def extract_group_values(data_group):
    """
    Extract metric names and paired values from a data dictionary.

    Parameters
    ----------
    data_group : dict
        Dictionary in which each key is a metric name and each value is a
        two-element list:
            [π-MSNet value, baseline value]

    Returns
    -------
    tuple
        A tuple containing:
            - metrics : list of str
                Metric names.
            - msnet_values : list of int or float
                Values corresponding to π-MSNet.
            - baseline_values : list of int or float
                Values corresponding to the baseline dataset.
    """

    metrics = list(data_group.keys())
    msnet_values = [data_group[metric][0] for metric in metrics]
    baseline_values = [data_group[metric][1] for metric in metrics]

    return metrics, msnet_values, baseline_values


def plot_grouped_bar_panel(
    axis,
    data_group,
    show_ylabel=False,
    use_scientific_yaxis=True,
    value_annotation_fontsize=15,
    increase_annotation_fontsize=15
):
    """
    Plot a grouped bar chart panel for metrics containing multiple categories.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis object on which the panel is drawn.
    data_group : dict
        Dictionary containing metric names and paired values.
    show_ylabel : bool, optional
        Whether to show the y-axis label "Count".
    use_scientific_yaxis : bool, optional
        Whether to format the y-axis in scientific notation.
    value_annotation_fontsize : int, optional
        Font size for absolute value annotations.
    increase_annotation_fontsize : int, optional
        Font size for relative increase annotations.

    Returns
    -------
    tuple
        Handles for the π-MSNet bars and baseline bars.
    """

    metrics, msnet_values, baseline_values = extract_group_values(data_group)

    x_positions = np.arange(len(metrics))

    msnet_bars = axis.bar(
        x_positions - GROUPED_BAR_WIDTH / 2,
        msnet_values,
        GROUPED_BAR_WIDTH,
        color=MSNET_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        label=MSNET_LABEL
    )

    baseline_bars = axis.bar(
        x_positions + GROUPED_BAR_WIDTH / 2,
        baseline_values,
        GROUPED_BAR_WIDTH,
        color=NINE_SPECIES_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        label=NINE_SPECIES_LABEL
    )

    if show_ylabel:
        axis.set_ylabel("Count")

    axis.set_xticks(x_positions)
    axis.set_xticklabels(metrics)

    if use_scientific_yaxis:
        set_scientific_yaxis(axis)

    apply_common_axis_style(axis)

    annotate_bar_values(
        axis,
        msnet_bars,
        msnet_values,
        offset_ratio=0.010,
        fontsize=value_annotation_fontsize
    )
    annotate_bar_values(
        axis,
        baseline_bars,
        baseline_values,
        offset_ratio=0.010,
        fontsize=value_annotation_fontsize
    )
    annotate_relative_increase(
        axis,
        baseline_bars,
        msnet_values,
        baseline_values,
        offset_ratio=0.1,
        fontsize=increase_annotation_fontsize
    )

    max_value = max(msnet_values + baseline_values)
    axis.set_ylim(0, max_value * TOP_MARGIN_RATIO)

    return msnet_bars, baseline_bars


def plot_single_metric_panel(
    axis,
    data_group,
    value_annotation_fontsize=15,
    increase_annotation_fontsize=15
):
    """
    Plot a grouped bar chart panel for a single metric.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis object on which the panel is drawn.
    data_group : dict
        Dictionary containing a single metric and paired values.
    value_annotation_fontsize : int, optional
        Font size for absolute value annotations.
    increase_annotation_fontsize : int, optional
        Font size for relative increase annotations.

    Returns
    -------
    tuple
        Handles for the π-MSNet bars and baseline bars.
    """

    metrics, msnet_values, baseline_values = extract_group_values(data_group)

    x_positions = np.array([0.0])

    msnet_bars = axis.bar(
        x_positions - SINGLE_PANEL_BAR_OFFSET,
        msnet_values,
        SINGLE_PANEL_BAR_WIDTH,
        color=MSNET_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        label=MSNET_LABEL
    )

    baseline_bars = axis.bar(
        x_positions + SINGLE_PANEL_BAR_OFFSET,
        baseline_values,
        SINGLE_PANEL_BAR_WIDTH,
        color=NINE_SPECIES_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        label=NINE_SPECIES_LABEL
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(metrics)
    axis.set_xlim(-0.45, 0.45)

    apply_common_axis_style(axis)

    annotate_bar_values(
        axis,
        msnet_bars,
        msnet_values,
        offset_ratio=0.030,
        fontsize=value_annotation_fontsize
    )
    annotate_bar_values(
        axis,
        baseline_bars,
        baseline_values,
        offset_ratio=0.030,
        fontsize=value_annotation_fontsize
    )
    annotate_relative_increase(
        axis,
        baseline_bars,
        msnet_values,
        baseline_values,
        offset_ratio=0.1,
        fontsize=increase_annotation_fontsize
    )

    max_value = max(msnet_values + baseline_values)
    axis.set_ylim(0, max_value * TOP_MARGIN_RATIO)

    return msnet_bars, baseline_bars


# =============================================================================
# Generate Figure 5b
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)

# Panel 1: Data scale and sequence coverage.
plot_grouped_bar_panel(
    axes[0],
    DATA_GROUP_1,
    show_ylabel=True,
    use_scientific_yaxis=True
)

# Panel 2: PTM information richness.
plot_grouped_bar_panel(
    axes[1],
    DATA_GROUP_2,
    show_ylabel=False,
    use_scientific_yaxis=True
)

# Panel 3: Species coverage breadth.
plot_single_metric_panel(
    axes[2],
    DATA_GROUP_3
)


# =============================================================================
# Add shared legend and export figure
# =============================================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1)
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    OUTPUT_FIGURE_FILE,
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)

plt.show()
