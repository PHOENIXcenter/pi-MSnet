# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:37:42 2025

@author: FineLiu

Convert peptide RT data from Parquet files into HDF5 shards,
then compute per-peptide statistics.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import h5py
import warnings
from collections import defaultdict
import json
warnings.filterwarnings('ignore')

def process_and_save_to_hdf5_sharded(root_dir, output_dir='peptide_shards',
                                     shard_size=50000, max_files_per_shard=1000):
    """
    Shard peptide data by first letter of peptide name and save into multiple HDF5 files.
    Reads Parquet files from root_dir/*/peptidoform_retention/*.parquet.
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Collect all peptide data, grouped by first letter
    peptide_shards = defaultdict(lambda: defaultdict(list))
    file_count = 0
    shard_index = {}  # mapping from peptide to shard key

    # Find all matching Parquet files
    parquet_files = list(root_path.glob('*/peptidoform_retention/*.parquet'))

    print(f"Found {len(parquet_files)} Parquet files")

    # Phase 1: collect data and group by first letter
    for i, parquet_file in enumerate(parquet_files):
        if (i + 1) % 100 == 0:
            print(f"Processing file {i + 1}: {parquet_file}")

        try:
            # Read Parquet file
            df = pd.read_parquet(parquet_file)

            # Check for required columns
            required_cols = ['peptidoform', 'retention', 'effective_rt']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: file {parquet_file} missing columns: {missing_cols}")
                continue

            # Compute normalized RT value
            df['adjust_rt'] = (df['retention'] / df['effective_rt']) * 100

            # Collect per-peptide data
            for peptidoform, group in df.groupby('peptidoform'):
                adjust_rts = group['adjust_rt'].values

                # Determine shard key (first letter, uppercase)
                if peptidoform:
                    shard_key = peptidoform[0].upper()
                    if not shard_key.isalpha():
                        shard_key = 'OTHER'
                else:
                    shard_key = 'OTHER'

                # Append to corresponding shard
                peptide_shards[shard_key][peptidoform].extend(adjust_rts)
                shard_index[peptidoform] = shard_key

            file_count += 1

        except Exception as e:
            print(f"Error processing file {parquet_file}: {str(e)}")
            continue

    print(f"Successfully processed {file_count} files, collected {len(shard_index)} unique peptides")
    print(f"Shard distribution: { {k: len(v) for k, v in peptide_shards.items()} }")

    # Phase 2: save each shard to its own HDF5 file
    print(f"Saving shards to directory: {output_dir}")

    # Save shard index mapping
    with open(output_path / 'shard_index.json', 'w') as f:
        json.dump(shard_index, f)

    # Process each shard
    for shard_key, peptides in peptide_shards.items():
        print(f"Processing shard '{shard_key}', containing {len(peptides)} peptides")

        # If shard is too large, split further
        if len(peptides) > shard_size:
            print(f"  Shard '{shard_key}' is too large ({len(peptides)} peptides), performing sub-sharding")
            _save_large_shard(peptides, shard_key, output_path, shard_size, max_files_per_shard)
        else:
            # Save single shard file
            shard_filename = output_path / f"peptide_rt_data_{shard_key}.h5"
            _save_shard(peptides, shard_filename)

    print("All peptide data successfully saved to shard files")
    return peptide_shards, shard_index

def _save_shard(peptides, filename):
    """Save a single shard to an HDF5 file."""
    with h5py.File(filename, 'w') as h5f:
        peptides_group = h5f.create_group('peptides')

        for peptidoform, rt_values in peptides.items():
            safe_name = make_safe_dataset_name(peptidoform)

            dataset = peptides_group.create_dataset(
                safe_name,
                data=np.array(rt_values, dtype=np.float32),
                compression="gzip",
                compression_opts=9
            )
            dataset.attrs['original_name'] = peptidoform
            dataset.attrs['count'] = len(rt_values)

    print(f"  Saved shard: {filename.name}")

def _save_large_shard(peptides, shard_key, output_path, shard_size, max_files_per_shard):
    """Split an oversized shard into multiple sub-shards."""
    peptide_items = list(peptides.items())
    total_peptides = len(peptide_items)

    # Calculate number of sub-shards needed
    num_subshards = min((total_peptides + shard_size - 1) // shard_size, max_files_per_shard)

    for i in range(num_subshards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, total_peptides)

        subshard_peptides = dict(peptide_items[start_idx:end_idx])
        subshard_filename = output_path / f"peptide_rt_data_{shard_key}_{i:03d}.h5"

        _save_shard(subshard_peptides, subshard_filename)

        print(f"    Sub-shard {i}: {start_idx}-{end_idx} ({len(subshard_peptides)} peptides)")

def make_safe_dataset_name(peptidoform):
    """Create a safe HDF5 dataset name."""
    safe_name = peptidoform.replace('/', '_').replace('\\', '_').replace(' ', '_')
    safe_name = safe_name.replace(':', '_').replace(';', '_').replace('|', '_')
    safe_name = safe_name.replace('*', '_').replace('?', '_').replace('"', '_')
    safe_name = safe_name.replace('<', '_').replace('>', '_')

    # If name too long, use MD5 hash
    if len(safe_name) > 150:
        import hashlib
        safe_name = hashlib.md5(peptidoform.encode()).hexdigest()

    return safe_name

def calculate_peptide_statistics_from_shards(shard_dir='peptide_shards'):
    """Compute per-peptide statistics from HDF5 shard files."""
    shard_path = Path(shard_dir)
    statistics_list = []

    # Get all shard files
    shard_files = list(shard_path.glob('peptide_rt_data_*.h5'))
    print(f"Found {len(shard_files)} shard files")

    for i, shard_file in enumerate(shard_files):
        if (i + 1) % 10 == 0:
            print(f"Processing shard file {i + 1}: {shard_file.name}")

        with h5py.File(shard_file, 'r') as h5f:
            peptides_group = h5f['peptides']

            for dataset_name in peptides_group:
                dataset = peptides_group[dataset_name]
                rt_values = dataset[:]
                n = len(rt_values)

                # Get original peptide name
                if 'original_name' in dataset.attrs:
                    peptidoform = dataset.attrs['original_name']
                else:
                    peptidoform = dataset_name

                if n < 2:
                    stats_dict = {
                        'peptidoform': peptidoform,
                        'count': n,
                        'mean': np.mean(rt_values) if n > 0 else np.nan,
                        'std': np.nan,
                        'median': rt_values[0] if n > 0 else np.nan,
                        'q5': rt_values[0] if n > 0 else np.nan,
                        'q25': rt_values[0] if n > 0 else np.nan,
                        'q75': rt_values[0] if n > 0 else np.nan,
                        'q95': rt_values[0] if n > 0 else np.nan,
                        'iqr': 0 if n > 0 else np.nan,
                        'min': rt_values[0] if n > 0 else np.nan,
                        'max': rt_values[0] if n > 0 else np.nan
                    }
                else:
                    mean_val = np.mean(rt_values)
                    std_val = np.std(rt_values, ddof=1)
                    median_val = np.median(rt_values)

                    q5, q25, q75, q95 = np.percentile(rt_values, [5, 25, 75, 95])
                    iqr_val = q75 - q25

                    min_val = np.min(rt_values)
                    max_val = np.max(rt_values)

                    stats_dict = {
                        'peptidoform': peptidoform,
                        'count': n,
                        'mean': mean_val,
                        'std': std_val,
                        'median': median_val,
                        'q5': q5,
                        'q25': q25,
                        'q75': q75,
                        'q95': q95,
                        'iqr': iqr_val,
                        'min': min_val,
                        'max': max_val
                    }

                statistics_list.append(stats_dict)

    # Convert to DataFrame
    summary_df = pd.DataFrame(statistics_list)
    return summary_df

def calculate_peptide_statistics_parallel(shard_dir='peptide_shards', num_processes=None):
    """
    Compute per-peptide statistics using multiprocessing for large datasets.
    """
    import multiprocessing as mp
    from functools import partial

    if num_processes is None:
        num_processes = mp.cpu_count()

    shard_path = Path(shard_dir)
    shard_files = list(shard_path.glob('peptide_rt_data_*.h5'))

    print(f"Using {num_processes} processes to compute statistics for {len(shard_files)} shard files")

    # Split tasks
    chunk_size = len(shard_files) // num_processes + 1
    chunks = [shard_files[i:i+chunk_size] for i in range(0, len(shard_files), chunk_size)]

    # Create partial function
    calc_chunk_func = partial(calculate_statistics_from_shard_files)

    # Use process pool
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(calc_chunk_func, chunks)

    # Merge results
    all_statistics = []
    for result in results:
        all_statistics.extend(result)

    return pd.DataFrame(all_statistics)

def calculate_statistics_from_shard_files(shard_files):
    """Compute statistics for a list of HDF5 shard files."""
    statistics_list = []

    for shard_file in shard_files:
        with h5py.File(shard_file, 'r') as h5f:
            peptides_group = h5f['peptides']

            for dataset_name in peptides_group:
                dataset = peptides_group[dataset_name]
                rt_values = dataset[:]
                n = len(rt_values)

                # Get original peptide name
                if 'original_name' in dataset.attrs:
                    peptidoform = dataset.attrs['original_name']
                else:
                    peptidoform = dataset_name

                if n < 2:
                    stats_dict = {
                        'peptidoform': peptidoform,
                        'count': n,
                        'mean': np.mean(rt_values) if n > 0 else np.nan,
                        'std': np.nan,
                        'median': rt_values[0] if n > 0 else np.nan,
                        'q5': rt_values[0] if n > 0 else np.nan,
                        'q25': rt_values[0] if n > 0 else np.nan,
                        'q75': rt_values[0] if n > 0 else np.nan,
                        'q95': rt_values[0] if n > 0 else np.nan,
                        'iqr': 0 if n > 0 else np.nan,
                        'min': rt_values[0] if n > 0 else np.nan,
                        'max': rt_values[0] if n > 0 else np.nan
                    }
                else:
                    mean_val = np.mean(rt_values)
                    std_val = np.std(rt_values, ddof=1)
                    median_val = np.median(rt_values)
                    q5, q25, q75, q95 = np.percentile(rt_values, [5, 25, 75, 95])
                    iqr_val = q75 - q25
                    min_val = np.min(rt_values)
                    max_val = np.max(rt_values)

                    stats_dict = {
                        'peptidoform': peptidoform,
                        'count': n,
                        'mean': mean_val,
                        'std': std_val,
                        'median': median_val,
                        'q5': q5,
                        'q25': q25,
                        'q75': q75,
                        'q95': q95,
                        'iqr': iqr_val,
                        'min': min_val,
                        'max': max_val
                    }

                statistics_list.append(stats_dict)

    return statistics_list

def main_sharded():
    root_dir = 'huizong'
    shard_dir = 'peptide_shards'
    summary_filename = 'peptide_rt_statistics_summary.csv'

    # 1. Process all Parquet files and save into sharded HDF5 files
    print("Processing peptide data and saving to sharded HDF5 files...")
    peptide_data, shard_index = process_and_save_to_hdf5_sharded(root_dir, shard_dir)

    if not peptide_data:
        print("No data found, exiting")
        return

    # 2. Compute statistics from shard files
    print("Computing statistics from shard files...")

    # Choose calculation method based on data volume
    total_peptides = sum(len(peptides) for peptides in peptide_data.values())
    if total_peptides > 100000:
        print("Large dataset detected, using parallel computation...")
        summary_df = calculate_peptide_statistics_parallel(shard_dir)
    else:
        summary_df = calculate_peptide_statistics_from_shards(shard_dir)

    # 3. Save statistics to CSV
    summary_df.to_csv(summary_filename, index=False)
    print(f"Statistics saved to: {summary_filename}")

    # 4. Display summary information
    print("\nData overview:")
    print(f"Total peptides: {len(summary_df)}")
    print("Observation count statistics:")
    print(summary_df['count'].describe())

if __name__ == "__main__":
    main_sharded()