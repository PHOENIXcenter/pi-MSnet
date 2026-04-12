from torch.utils.data import IterableDataset
import duckdb
import numpy as np


class RTIterableDataset(IterableDataset):

    def __init__(self, parquet_path, batch_size=100_000,
                 min_consensus_support=None,
                 max_pep=None
                 ):
        con = duckdb.connect()

        query = f"""
            SELECT
                peptidoform,
                retention_time,
                consensus_support,
                posterior_error_probability
            FROM parquet_scan('{parquet_path}')
            ORDER BY length(sequence)
        """

        self.arrow_table = con.execute(query).to_arrow_table()
        self.batch_size = batch_size
        self.min_consensus_support = min_consensus_support
        self.max_pep = max_pep

    def __iter__(self):
        for batch in self.arrow_table.to_batches(max_chunksize=self.batch_size):
            yield self.process_batch(batch)

    def process_batch(self, batch):
        peptidoform = batch["peptidoform"].to_pylist()
        retention_time = batch["retention_time"].to_pylist()
        rt = np.array(retention_time, dtype=np.float32) / 60.0

        return {
            "peptide": peptidoform,
            "rt": rt
        }

    # =========================================================
    # Optional FILTER 1
    # =========================================================
    def filter_by_consensus_support(self, support):
        """
        Keep spectrum if consensus_support >= threshold
        """
        if self.min_consensus_support is None:
            return True

        if support is None:
            return False

        return support >= self.min_consensus_support

    # =========================================================
    # Optional FILTER 2
    # =========================================================
    def filter_by_pep(self, pep):
        """
        Keep spectrum if posterior_error_probability <= threshold
        """
        if self.max_pep is None:
            return True

        if pep is None:
            return False

        return pep <= self.max_pep
