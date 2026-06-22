# -*- coding: utf-8 -*-
"""
HDF5 to DuckDB Conversion Script
Function: Convert peptide RT data from HDF5 shards into a DuckDB database
with concise progress reporting and high-performance batch insertion.
"""

import h5py
import duckdb
import pathlib
import tqdm
import os
import pandas as pd
import numpy as np
import csv
from datetime import datetime
import sys
import gc
import time
import traceback

# ========== Configuration ==========
H5_DIR = pathlib.Path("peptide_shards")    # Directory containing HDF5 shard files
DB_FILE = "peptide_rt.duckdb"             # Output DuckDB database file
LOG_CSV = "filtered_records_log.csv"      # Filtered records log
BATCH_SZ = 100000                         # Batch size (adjust based on memory)
ENABLE_INDEX = True                       # Create indexes after insertion
SHOW_PROGRESS_BAR = True                  # Display progress bar

# ========== Filter Logger ==========
class FilterLogger:
    """Logs filtered-out records to a CSV file."""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.log_count = 0
        self.csv_file = open(log_file, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'h5_file', 'dataset', 'index',
            'original_value', 'reason', 'peptidoform'
        ])
    
    def log(self, h5_file, dataset_name, index, original_value, reason, peptidoform=None):
        """Record a single filtered record."""
        self.log_count += 1
        h5_filename = os.path.basename(h5_file) if h5_file else "unknown"
        self.csv_writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            h5_filename,
            dataset_name,
            index,
            str(original_value)[:100],
            reason,
            str(peptidoform)[:100] if peptidoform else ""
        ])
        if self.log_count % 10000 == 0:
            self.csv_file.flush()
    
    def close(self):
        """Close the log file."""
        if self.csv_file:
            self.csv_file.close()

# ========== Data Generator ==========
def iter_h5_records(h5_files, filter_logger=None):
    """Generator that yields (peptidoform, rt_value) from HDF5 shards."""
    for h5file in h5_files:
        try:
            with h5py.File(h5file, "r") as h:
                # Locate the peptides group
                peptides_group = None
                if "peptides" in h:
                    peptides_group = h["peptides"]
                else:
                    for key in h.keys():
                        if 'peptide' in key.lower():
                            peptides_group = h[key]
                            break
                if peptides_group is None:
                    continue
                
                # Iterate over datasets
                for ds_name, ds in peptides_group.items():
                    peptidoform = ds.attrs.get("original_name", ds_name)
                    try:
                        data_array = ds[:]
                        for i, val in enumerate(data_array):
                            # Data cleaning
                            if val is None:
                                if filter_logger:
                                    filter_logger.log(h5file.name, ds_name, i, val, "NULL_VALUE", peptidoform)
                                continue
                            try:
                                rt_float = float(val)
                            except (ValueError, TypeError):
                                if filter_logger:
                                    filter_logger.log(h5file.name, ds_name, i, val, "CONVERSION_ERROR", peptidoform)
                                continue
                            if np.isnan(rt_float):
                                if filter_logger:
                                    filter_logger.log(h5file.name, ds_name, i, val, "NaN_VALUE", peptidoform)
                                continue
                            if np.isinf(rt_float):
                                if filter_logger:
                                    filter_logger.log(h5file.name, ds_name, i, val, "INFINITE_VALUE", peptidoform)
                                continue
                            yield peptidoform, rt_float
                    except Exception:
                        continue
        except Exception:
            continue

# ========== High-Performance Batch Inserter ==========
class HighPerformanceBatchInserter:
    """Batch inserter using Pandas DataFrame for high-speed DuckDB inserts."""
    
    def __init__(self, db_file, batch_size=100000):
        self.db_file = db_file
        self.batch_size = batch_size
        self.batch = []
        self.processed_count = 0
        self.inserted_count = 0
        self.conn = duckdb.connect(db_file)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS peptide_rt (
                peptidoform VARCHAR NOT NULL,
                rt_value   DOUBLE  NOT NULL
            )
        """)
    
    def add_record(self, peptidoform, rt_value):
        """Add a record to the current batch; flush if batch is full."""
        self.batch.append((peptidoform, rt_value))
        self.processed_count += 1
        if len(self.batch) >= self.batch_size:
            self._insert_batch()
    
    def _insert_batch(self):
        """Insert the current batch into the database."""
        if not self.batch:
            return
        try:
            df = pd.DataFrame(self.batch, columns=["peptidoform", "rt_value"])
            self.conn.register("tmp_batch", df)
            self.conn.execute("INSERT INTO peptide_rt SELECT * FROM tmp_batch")
            inserted = len(self.batch)
            self.inserted_count += inserted
            self.batch.clear()
            if self.inserted_count % 1000000 == 0:
                self.conn.commit()
        except Exception as e:
            print(f"\nBatch insertion failed: {e}")
            if self.batch:
                print(f"Failed batch size: {len(self.batch)}")
                try:
                    df_check = pd.DataFrame(self.batch, columns=["peptidoform", "rt_value"])
                    print("DataFrame info:")
                    print(f"  Shape: {df_check.shape}")
                    print(f"  Column types: {df_check.dtypes.to_dict()}")
                    print(f"  Has NaN: {df_check.isna().any().any()}")
                except Exception as df_e:
                    print(f"DataFrame check failed: {df_e}")
            raise
    
    def flush(self):
        """Insert any remaining records in the batch."""
        self._insert_batch()
    
    def close(self):
        """Flush remaining records, create indexes, and close connection."""
        if self.batch:
            try:
                self._insert_batch()
            except Exception as e:
                print(f"\nFlushing final batch failed: {e}")
        if self.conn:
            self.conn.commit()
            if ENABLE_INDEX:
                print("\nCreating indexes...")
                try:
                    self.conn.execute("CREATE INDEX IF NOT EXISTS idx_peptidoform ON peptide_rt(peptidoform)")
                    self.conn.execute("CREATE INDEX IF NOT EXISTS idx_rt_value ON peptide_rt(rt_value)")
                except Exception as e:
                    print(f"Index creation failed: {e}")
            self.conn.close()

# ========== Main Function ==========
def main():
    """Main conversion routine."""
    print("=" * 60)
    print("HDF5 to DuckDB Converter")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if not H5_DIR.exists() or not H5_DIR.is_dir():
        print(f"Error: Directory {H5_DIR} does not exist or is not a directory.")
        sys.exit(1)
    
    h5_files = list(H5_DIR.glob("peptide_rt_data_*.h5"))
    total_h5_files = len(h5_files)
    if total_h5_files == 0:
        print(f"Error: No peptide_rt_data_*.h5 files found in {H5_DIR}.")
        sys.exit(1)
    print(f"Found {total_h5_files} HDF5 file(s).")
    
    # Remove existing database and log files
    if os.path.exists(DB_FILE):
        print(f"Removing existing database: {DB_FILE}")
        try:
            os.remove(DB_FILE)
        except Exception as e:
            print(f"Failed to remove database file: {e}")
            sys.exit(1)
    if os.path.exists(LOG_CSV):
        try:
            os.remove(LOG_CSV)
        except Exception:
            pass
    
    filter_logger = FilterLogger(LOG_CSV)
    inserter = HighPerformanceBatchInserter(DB_FILE, BATCH_SZ)
    start_time = time.time()
    last_report_time = start_time
    
    print(f"Batch size: {BATCH_SZ:,} records")
    print("Starting data import...\n")
    
    try:
        if SHOW_PROGRESS_BAR:
            pbar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            with tqdm.tqdm(
                desc="Import progress",
                unit="rec",
                bar_format=pbar_format,
                mininterval=1.0,
                maxinterval=5.0,
                smoothing=0.1
            ) as pbar:
                for i, (pep, rt) in enumerate(iter_h5_records(h5_files, filter_logger)):
                    inserter.add_record(pep, rt)
                    pbar.update(1)
                    if i % 10000 == 0:
                        current_time = time.time()
                        elapsed = current_time - start_time
                        speed = (i + 1) / elapsed if elapsed > 0 else 0
                        postfix_info = {
                            'speed': f"{speed:,.0f} rec/s",
                            'filtered': f"{filter_logger.log_count:,}",
                            'inserted': f"{inserter.inserted_count:,}"
                        }
                        pbar.set_postfix(postfix_info)
                    current_time = time.time()
                    if current_time - last_report_time > 30:
                        elapsed = current_time - start_time
                        hours = elapsed / 3600
                        speed = (i + 1) / elapsed if elapsed > 0 else 0
                        tqdm.tqdm.write(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"Processed: {i+1:,} records, "
                            f"Speed: {speed:,.0f} rec/s, "
                            f"Elapsed: {hours:.2f} h"
                        )
                        last_report_time = current_time
                        if (i + 1) % 1000000 == 0:
                            gc.collect()
                tqdm.tqdm.write("\nData read complete. Flushing remaining batch...")
                inserter.flush()
        else:
            print("Processing without progress bar...")
            record_count = 0
            for i, (pep, rt) in enumerate(iter_h5_records(h5_files, filter_logger)):
                inserter.add_record(pep, rt)
                record_count += 1
                if record_count % 100000 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    speed = record_count / elapsed if elapsed > 0 else 0
                    hours = elapsed / 3600
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Processed: {record_count:,} records, "
                        f"Speed: {speed:,.0f} rec/s, "
                        f"Elapsed: {hours:.2f} h, "
                        f"Filtered: {filter_logger.log_count:,}"
                    )
                    if record_count % 1000000 == 0:
                        gc.collect()
            print("\nData read complete. Flushing remaining batch...")
            inserter.flush()
            
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        print("Saving processed data...")
        try:
            inserter.flush()
            print("Remaining batch inserted.")
        except Exception as e:
            print(f"Flush failed: {e}")
        print("\nProgress snapshot:")
        print(f"  Processed: {inserter.processed_count:,}")
        print(f"  Inserted:  {inserter.inserted_count:,}")
        print(f"  Filtered:  {filter_logger.log_count:,}")
        inserter.close()
        filter_logger.close()
        print(f"\nDatabase saved: {DB_FILE}")
        print(f"Filter log saved: {LOG_CSV}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        traceback.print_exc()
        print("\nAttempting to save processed data...")
        try:
            inserter.flush()
            print(f"Saved {inserter.inserted_count:,} records to database.")
        except Exception as flush_e:
            print(f"Failed to save data: {flush_e}")
        print("\nProgress snapshot:")
        print(f"  Processed: {inserter.processed_count:,}")
        print(f"  Inserted:  {inserter.inserted_count:,}")
        print(f"  Filtered:  {filter_logger.log_count:,}")
        inserter.close()
        filter_logger.close()
        sys.exit(1)
    
    inserter.close()
    filter_logger.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Gather final statistics
    try:
        conn = duckdb.connect(DB_FILE)
        stats = conn.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT peptidoform) as unique_peptides,
                MIN(rt_value) as min_rt,
                MAX(rt_value) as max_rt,
                AVG(rt_value) as avg_rt,
                STDDEV_POP(rt_value) as std_rt
            FROM peptide_rt
        """).fetchone()
        conn.close()
    except Exception as e:
        print(f"Failed to retrieve database statistics: {e}")
        stats = (inserter.inserted_count, 0, 0.0, 0.0, 0.0, 0.0)
    
    # Final report
    print("\n" + "=" * 60)
    print("Processing completed!")
    print("=" * 60)
    hours = elapsed_time / 3600
    print("Timing:")
    print(f"  Elapsed: {elapsed_time:.2f} s ({hours:.2f} h)")
    print(f"  Average speed: {inserter.processed_count/elapsed_time:,.0f} rec/s")
    print("\nData statistics:")
    print(f"  Total processed: {inserter.processed_count:,}")
    print(f"  Valid inserted:  {stats[0]:,}")
    print(f"  Filtered out:    {filter_logger.log_count:,}")
    if inserter.processed_count > 0:
        print(f"  Efficiency: {stats[0]/inserter.processed_count*100:.2f}%")
    print(f"  Unique peptides: {stats[1]:,}")
    print(f"  RT range: {stats[2]:.4f} - {stats[3]:.4f}")
    print("\nOutput files:")
    db_size = os.path.getsize(DB_FILE) / (1024 * 1024)  # MB
    log_size = os.path.getsize(LOG_CSV) / 1024 if os.path.exists(LOG_CSV) else 0  # KB
    print(f"  Database: {DB_FILE} ({db_size:.2f} MB)")
    print(f"  Filter log: {LOG_CSV} ({log_size:.2f} KB)")
    
    summary_file = f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("HDF5 to DuckDB Processing Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input directory: {H5_DIR}\n")
        f.write(f"HDF5 files: {total_h5_files}\n")
        f.write(f"Total processed: {inserter.processed_count}\n")
        f.write(f"Valid inserted: {stats[0]}\n")
        f.write(f"Filtered out: {filter_logger.log_count}\n")
        if inserter.processed_count > 0:
            f.write(f"Efficiency: {stats[0]/inserter.processed_count*100:.2f}%\n")
        f.write(f"Unique peptides: {stats[1]}\n")
        f.write(f"Elapsed time: {elapsed_time:.2f} s ({hours:.2f} h)\n")
        f.write(f"Average speed: {inserter.processed_count/elapsed_time:,.0f} rec/s\n")
        f.write(f"Database size: {db_size:.2f} MB\n")
    
    print(f"\nSummary report saved: {summary_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()