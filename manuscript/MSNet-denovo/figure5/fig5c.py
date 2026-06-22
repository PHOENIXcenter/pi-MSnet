#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Title:
    Figure 5c - Peptide length distribution in the dataset

Description:
    This script generates Figure 5c for the manuscript. It reads peptide
    precursor information, removes mass-shift modification annotations from
    peptide sequences, calculates peptide length distributions after modification
    stripping, summarizes modification occurrence statistics, computes peptide
    length diversity metrics, and visualizes the peptide length distribution as
    a bar chart.

    Modification annotations are identified using the following regular
    expression pattern:

        +<digits>.<digits>

    For example, peptide strings containing mass-shift annotations such as
    "+15.9949" or "+79.9663" will be stripped before peptide length calculation.

Input:
    - fig5c_precursors.csv:
        Precursor table containing peptide sequence information.
        Required columns:
            - Titles
            - Peptides
            - Charges

Output:
    - peptide_length.svg:
        Publication-ready bar chart showing the peptide length distribution.

    - Console output:
        Modification statistics, stripped peptide statistics, peptide length
        distribution, and length diversity metrics.

Author:
    Tianze Ling, Ph.D. candidate @ Tsinghua University and 
    National Center for Protein Sciences (Beijing)

Contact:
    tianzeling98@outlook.com

Usage:
    python fig5c.py

Notes:
    - Peptide length is calculated after removing mass-shift modification
      annotations.
    - Duplicate peptide entries are retained when calculating the length
      distribution, consistent with the original script.
    - Unique stripped peptide counts are reported separately.
===============================================================================
"""

import math
import re
import statistics
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

INPUT_FILE = "fig5c_precursors.csv"
OUTPUT_FIGURE_FILE = "peptide_length.svg"

PEPTIDE_COLUMN = "Peptides"

# Match mass-shift modification annotations, e.g., +15.9949 or +79.9663.
MODIFICATION_PATTERN = re.compile(r"\+\d+\.\d+")

FIGURE_SIZE = (8, 5)
FIGURE_DPI = 300
BAR_COLOR = "#5B9BBF"


# =============================================================================
# Global plotting style
# =============================================================================

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.titlesize"] = 15
mpl.rcParams["axes.labelsize"] = 13
mpl.rcParams["xtick.labelsize"] = 11
mpl.rcParams["ytick.labelsize"] = 11
mpl.rcParams["legend.fontsize"] = 11
mpl.rcParams["figure.titlesize"] = 16
mpl.rcParams["axes.unicode_minus"] = False


# =============================================================================
# Helper functions
# =============================================================================

def load_peptides(input_file, peptide_column):
    """
    Load peptide sequences from a CSV file.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file.
    peptide_column : str
        Name of the column containing peptide sequences.

    Returns
    -------
    list of str
        Peptide sequences extracted from the specified column.
    """

    precursor_table = pd.read_csv(input_file)

    if peptide_column not in precursor_table.columns:
        raise ValueError(
            f"Required column '{peptide_column}' was not found in "
            f"the input file: {input_file}"
        )

    peptides = list(precursor_table[peptide_column])

    return peptides


def strip_modifications_and_count(peptides, modification_pattern):
    """
    Remove mass-shift modification annotations and count modification types.

    Parameters
    ----------
    peptides : list of str
        Original peptide sequences, potentially containing modification
        annotations.
    modification_pattern : re.Pattern
        Compiled regular expression pattern used to identify modifications.

    Returns
    -------
    tuple
        A tuple containing:
            - stripped_peptides : list of str
                Peptide sequences after removing modification annotations.
            - modification_peptide_count : collections.Counter
                Number of peptide entries containing each modification type.
                A modification is counted only once per peptide entry.
            - modification_total_count : collections.Counter
                Total occurrence count of each modification type across all
                peptide entries.
    """

    modification_peptide_count = Counter()
    modification_total_count = Counter()
    stripped_peptides = []

    for peptide in peptides:
        peptide = str(peptide)

        modifications = modification_pattern.findall(peptide)
        unique_modifications = set(modifications)

        for modification in unique_modifications:
            modification_peptide_count[modification] += 1

        for modification in modifications:
            modification_total_count[modification] += 1

        stripped_peptide = modification_pattern.sub("", peptide)
        stripped_peptides.append(stripped_peptide)

    return stripped_peptides, modification_peptide_count, modification_total_count


def calculate_length_statistics(stripped_peptides):
    """
    Calculate peptide length distribution and diversity statistics.

    Parameters
    ----------
    stripped_peptides : list of str
        Peptide sequences after removing modification annotations.

    Returns
    -------
    dict
        Dictionary containing peptide length counts and summary statistics.
    """

    lengths = [len(peptide) for peptide in stripped_peptides]
    length_count = Counter(lengths)
    total_count = len(lengths)

    if total_count == 0:
        return {
            "lengths": lengths,
            "length_count": length_count,
            "total_peptides_after_strip": 0,
            "unique_peptides_after_strip": 0,
            "min_len": 0,
            "max_len": 0,
            "mean_len": 0,
            "median_len": 0,
            "richness": 0,
            "shannon_entropy": 0,
            "simpson_diversity": 0,
        }

    shannon_entropy = 0
    for count in length_count.values():
        probability = count / total_count
        shannon_entropy -= probability * math.log(probability)

    simpson_diversity = 1 - sum(
        (count / total_count) ** 2
        for count in length_count.values()
    )

    statistics_dict = {
        "lengths": lengths,
        "length_count": length_count,
        "total_peptides_after_strip": len(stripped_peptides),
        "unique_peptides_after_strip": len(set(stripped_peptides)),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "mean_len": statistics.mean(lengths),
        "median_len": statistics.median(lengths),
        "richness": len(length_count),
        "shannon_entropy": shannon_entropy,
        "simpson_diversity": simpson_diversity,
    }

    return statistics_dict


def print_summary(
    modification_peptide_count,
    modification_total_count,
    length_statistics
):
    """
    Print modification statistics and peptide length statistics to the console.

    Parameters
    ----------
    modification_peptide_count : collections.Counter
        Number of peptide entries containing each modification type.
    modification_total_count : collections.Counter
        Total occurrence count of each modification type.
    length_statistics : dict
        Dictionary containing peptide length statistics.

    Returns
    -------
    None
        The function prints summary information to the console.
    """

    print("=== Modification types and the number of corresponding peptide entries ===")
    for modification, count in modification_peptide_count.most_common():
        print(f"{modification}\t{count}")

    print("\n=== Total number of modification occurrences ===")
    for modification, count in modification_total_count.most_common():
        print(f"{modification}\t{count}")

    print("\n=== Peptide statistics after modification stripping ===")
    print(
        "Total number of stripped peptide entries, including duplicates: "
        f"{length_statistics['total_peptides_after_strip']}"
    )
    print(
        "Number of unique stripped peptides: "
        f"{length_statistics['unique_peptides_after_strip']}"
    )

    print("\n=== Peptide length distribution ===")
    for peptide_length, count in sorted(length_statistics["length_count"].items()):
        print(f"Length {peptide_length}: {count}")

    print("\n=== Peptide length diversity metrics ===")
    print(f"Minimum length: {length_statistics['min_len']}")
    print(f"Maximum length: {length_statistics['max_len']}")
    print(f"Mean length: {length_statistics['mean_len']:.2f}")
    print(f"Median length: {length_statistics['median_len']}")
    print(f"Length richness, number of distinct lengths: {length_statistics['richness']}")
    print(f"Shannon entropy: {length_statistics['shannon_entropy']:.4f}")
    print(f"Simpson diversity: {length_statistics['simpson_diversity']:.4f}")


def plot_peptide_length_distribution(length_count, output_file):
    """
    Plot and save the peptide length distribution.

    Parameters
    ----------
    length_count : collections.Counter
        Count of peptide entries for each peptide length.
    output_file : str
        Path to the output figure file.

    Returns
    -------
    None
        The function saves the figure and displays it.
    """

    sorted_lengths = sorted(length_count.keys())
    counts = [length_count[length] for length in sorted_lengths]

    plt.figure(figsize=FIGURE_SIZE)
    plt.bar(
        sorted_lengths,
        counts,
        color=BAR_COLOR
    )

    plt.xlabel("Peptide Length")
    plt.ylabel("Count")
    plt.title("Peptide Length Distribution")
    plt.tight_layout()
    plt.savefig(output_file, dpi=FIGURE_DPI)
    plt.show()


# =============================================================================
# Main workflow
# =============================================================================

peptide_sequences = load_peptides(
    input_file=INPUT_FILE,
    peptide_column=PEPTIDE_COLUMN
)

stripped_peptide_sequences, mod_peptide_count, mod_total_count = (
    strip_modifications_and_count(
        peptides=peptide_sequences,
        modification_pattern=MODIFICATION_PATTERN
    )
)

peptide_length_statistics = calculate_length_statistics(
    stripped_peptides=stripped_peptide_sequences
)

print_summary(
    modification_peptide_count=mod_peptide_count,
    modification_total_count=mod_total_count,
    length_statistics=peptide_length_statistics
)

plot_peptide_length_distribution(
    length_count=peptide_length_statistics["length_count"],
    output_file=OUTPUT_FIGURE_FILE
)
