# -*- coding: utf-8 -*-
"""
Peptide Retention Time Confidence Calculator
Function: Compute confidence scores for peptide RT predictions using DuckDB data.
"""

import duckdb
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

DB_FILE = "peptide_rt.duckdb"
SUMMARY_CSV = "peptide_rt_statistics_summary.csv"

class DuckDBConnectionManager:
    """Singleton class managing DuckDB connection."""
    _instance = None
    _conn = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_connection()
        return cls._instance

    def _initialize_connection(self):
        """Initialize database connection."""
        if not os.path.exists(DB_FILE):
            raise FileNotFoundError(
                f"Database file {DB_FILE} not found. Please run the data import script first."
            )
        try:
            self._conn = duckdb.connect(DB_FILE)
            # Verify database schema
            table_count = self._conn.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = 'peptide_rt'
            """).fetchone()[0]
            if table_count == 0:
                raise ValueError(f"Table 'peptide_rt' not found in database {DB_FILE}.")
            print(f"Connected to database: {DB_FILE}")
            print(f"Database size: {os.path.getsize(DB_FILE)/1024/1024:.2f} MB")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

    @property
    def conn(self):
        """Get database connection."""
        if self._conn is None:
            self._initialize_connection()
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()

# Global connection manager
_CONN_MANAGER = DuckDBConnectionManager()

def get_connection():
    """Return the shared DuckDB connection."""
    return _CONN_MANAGER.conn

def _get_rt_array(peptide_name: str):
    """Retrieve all RT values for a given peptide from the database."""
    conn = get_connection()
    count_result = conn.execute(
        "SELECT COUNT(*) FROM peptide_rt WHERE peptidoform = ?",
        [peptide_name]
    ).fetchone()[0]
    if count_result == 0:
        return np.array([])
    arr = conn.execute(
        "SELECT rt_value FROM peptide_rt WHERE peptidoform = ?",
        [peptide_name]
    ).fetchnumpy()["rt_value"]
    return arr

# ---------- Four confidence algorithms ----------
def method1_quantile_based(peptide_name, predicted_rt, summary_df, shard_dir=None, k=1.0):
    """Quantile-based confidence score."""
    rt_values = _get_rt_array(peptide_name)
    n = len(rt_values)
    if n < 2:
        return 0.5 if n == 1 else 0.3
    sorted_rt = np.sort(rt_values)
    idx = np.searchsorted(sorted_rt, predicted_rt)
    if 0 < idx < n:
        percentile = (idx - 1 + (predicted_rt - sorted_rt[idx-1]) /
                      (sorted_rt[idx] - sorted_rt[idx-1])) / n
    else:
        percentile = float(idx == n)
    deviation = abs(percentile - 0.5) * 2
    base = 1 - deviation
    uncertainty = k / np.sqrt(n)
    return max(0.0, min(1.0, base * (1 - uncertainty)))

def method2_robust_zscore(peptide_name, predicted_rt, summary_df, shard_dir=None, k=1.0):
    """Robust Z-score based confidence score."""
    row = summary_df[summary_df["peptidoform"] == peptide_name]
    if row.empty:
        return 0.3
    n, median, iqr, std = row.iloc[0][["count", "median", "iqr", "std"]]
    if n < 2:
        return 0.5
    spread = iqr / 1.349 if iqr > 0 else (std if pd.notna(std) and std > 0 else None)
    if spread is None:
        return 0.5
    robust_z = abs(predicted_rt - median) / spread
    if n > 2:
        p = 2 * (1 - stats.t.cdf(robust_z, df=n-1))
    else:
        p = 1 / (1 + robust_z)
    base = 1 - min(p, 1.0)
    uncertainty = k / np.sqrt(n)
    return max(0.0, min(1.0, base * (1 - uncertainty)))

def method3_distance_based(peptide_name, predicted_rt, summary_df, shard_dir=None, k=1.0, spread_factor=2.0):
    """Distance-based confidence score."""
    row = summary_df[summary_df["peptidoform"] == peptide_name]
    if row.empty:
        return 0.3
    n, median, iqr, std = row.iloc[0][["count", "median", "iqr", "std"]]
    if n < 2:
        return 0.5
    spread = iqr * spread_factor if iqr > 0 else (std if pd.notna(std) and std > 0 else None)
    if spread is None:
        return 0.5
    distance = abs(predicted_rt - median)
    if distance <= spread:
        base = max(0.0, 1 - distance / spread)
    else:
        base = 0.0
    uncertainty = k / np.sqrt(n)
    return max(0.0, min(1.0, base * (1 - uncertainty)))

def method4_kde_based(peptide_name, predicted_rt, summary_df, shard_dir=None, k=1.0):
    """Kernel Density Estimation based confidence score."""
    rt_values = _get_rt_array(peptide_name)
    n = len(rt_values)
    if n < 2:
        return 0.5 if n == 1 else 0.3
    try:
        kde = gaussian_kde(rt_values)
        pred_dens = kde(predicted_rt)[0]
        rt_range = np.linspace(rt_values.min(), rt_values.max(), 100)
        max_dens = kde(rt_range).max()
        base = pred_dens / max_dens if max_dens > 0 else 0.0
    except Exception:
        # Fallback to quantile method if KDE fails
        return method1_quantile_based(peptide_name, predicted_rt, summary_df, shard_dir, k)
    uncertainty = k / np.sqrt(n)
    return max(0.0, min(1.0, base * (1 - uncertainty)))

# ---------- Calculator class ----------
class ShardedPeptideConfidenceCalculator:
    """Peptide confidence calculator."""
    
    def __init__(self, summary_df_path=SUMMARY_CSV, shard_dir=None):
        """
        Initialize calculator.
        
        Parameters:
            summary_df_path: path to summary CSV file.
            shard_dir: (unused, kept for backward compatibility)
        """
        if not os.path.exists(summary_df_path):
            raise FileNotFoundError(
                f"Summary statistics file {summary_df_path} not found. Please run the statistics generation script first."
            )
        self.summary_df = pd.read_csv(summary_df_path)
        self.pep2idx = {row["peptidoform"]: i for i, row in self.summary_df.iterrows()}
        print("Peptide confidence calculator initialized.")
        print(f"  Loaded statistics for {len(self.summary_df)} peptides.")
        print(f"  Database file: {DB_FILE}")

    def get_peptide_data(self, peptide_name):
        """Retrieve all RT values for a peptide."""
        return _get_rt_array(peptide_name)

    def get_peptide_stats(self, peptide_name):
        """Retrieve statistics summary for a peptide."""
        if peptide_name in self.pep2idx:
            return self.summary_df.iloc[self.pep2idx[peptide_name]].to_dict()
        return None

    def calculate_all_confidences(self, peptide_name, predicted_rt):
        """
        Compute confidence scores using all four methods.
        
        Parameters:
            peptide_name: peptide identifier.
            predicted_rt: predicted retention time.
            
        Returns:
            Dictionary with method scores and average confidence.
        """
        if peptide_name not in self.pep2idx:
            return {
                'peptide': peptide_name,
                'predicted_rt': predicted_rt,
                'error': 'Peptide not found in summary',
                'method1_quantile': 0.3,
                'method2_robust_zscore': 0.3,
                'method3_distance': 0.3,
                'method4_kde': 0.3,
                'average_confidence': 0.3
            }
        try:
            c1 = method1_quantile_based(peptide_name, predicted_rt, self.summary_df)
            c2 = method2_robust_zscore(peptide_name, predicted_rt, self.summary_df)
            c3 = method3_distance_based(peptide_name, predicted_rt, self.summary_df)
            c4 = method4_kde_based(peptide_name, predicted_rt, self.summary_df)
            avg = np.mean([c1, c2, c3, c4])
            return {
                'peptide': peptide_name,
                'predicted_rt': predicted_rt,
                'method1_quantile': round(c1, 4),
                'method2_robust_zscore': round(c2, 4),
                'method3_distance': round(c3, 4),
                'method4_kde': round(c4, 4),
                'average_confidence': round(avg, 4)
            }
        except Exception as e:
            return {
                'peptide': peptide_name,
                'predicted_rt': predicted_rt,
                'error': f'Calculation error: {str(e)}',
                'method1_quantile': 0.3,
                'method2_robust_zscore': 0.3,
                'method3_distance': 0.3,
                'method4_kde': 0.3,
                'average_confidence': 0.3
            }

    def batch_calculate_confidences(self, peptide_rt_pairs):
        """
        Batch confidence calculation.
        
        Parameters:
            peptide_rt_pairs: list of (peptide_name, predicted_rt) tuples.
            
        Returns:
            DataFrame with confidence results.
        """
        results = []
        total = len(peptide_rt_pairs)
        for i, (peptide, rt) in enumerate(peptide_rt_pairs):
            if i > 0 and i % 100 == 0:
                print(f"  Processed {i}/{total} peptides...")
            result = self.calculate_all_confidences(peptide, rt)
            results.append(result)
        return pd.DataFrame(results)

    def validate_database(self):
        """Verify database connection and structure."""
        try:
            conn = get_connection()
            columns = conn.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'peptide_rt'
                ORDER BY ordinal_position
            """).fetchall()
            print("Database validation passed.")
            print("Table structure:")
            for col_name, col_type in columns:
                print(f"  {col_name}: {col_type}")
            indexes = conn.execute("""
                SELECT index_name
                FROM duckdb_indexes()
                WHERE table_name = 'peptide_rt'
            """).fetchall()
            if indexes:
                print("Indexes:")
                for idx in indexes:
                    print(f"  {idx[0]}")
            return True
        except Exception as e:
            print(f"Database validation failed: {e}")
            return False

# ---------- Usage example ----------
if __name__ == "__main__":
    try:
        calculator = ShardedPeptideConfidenceCalculator()
        calculator.validate_database()
        test_peptide = "EXAMPLE_PEPTIDE"   # Replace with actual peptide name
        test_rt = 10.5                      # Replace with actual predicted RT
        print(f"\nSingle peptide confidence calculation:")
        print(f"  Peptide: {test_peptide}")
        print(f"  Predicted RT: {test_rt}")
        result = calculator.calculate_all_confidences(test_peptide, test_rt)
        for key, value in result.items():
            print(f"  {key}: {value}")

        print(f"\nBatch calculation example:")
        test_pairs = [
            (test_peptide, test_rt),
            ("ANOTHER_PEPTIDE", 15.2),
        ]
        batch_results = calculator.batch_calculate_confidences(test_pairs)
        print(batch_results)

        print(f"\nRetrieve peptide statistics:")
        stats = calculator.get_peptide_stats(test_peptide)
        if stats:
            for key, value in stats.items():
                print(f"  {key}: {value}")

    except FileNotFoundError as e:
        print(f"Initialization failed: {e}")
        print(f"Please ensure the following files exist:")
        print(f"  1. Database file: {DB_FILE}")
        print(f"  2. Summary statistics file: {SUMMARY_CSV}")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()