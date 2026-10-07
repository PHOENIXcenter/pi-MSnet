import duckdb
import numpy as np
from torch.utils.data import IterableDataset

from msnetloader.utils import detect_parquet_schema, validate_batch_size


class RTIterableDataset(IterableDataset):

    def __init__(self, parquet_path, batch_size=32,
                 min_consensus_support=None,
                 max_pep=None,
                 extra_where=None
                 ):
        self.min_consensus_support = min_consensus_support
        self.max_pep = max_pep

        conditions = []
        params = []

        if self.min_consensus_support is not None:
            conditions.append("consensus_support >= ?")
            params.append(self.min_consensus_support)

        if self.max_pep is not None:
            conditions.append("posterior_error_probability <= ?")
            params.append(self.max_pep)

        if extra_where is not None:
            conditions.append(f"({extra_where})")

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        rt_col = "rt" if detect_parquet_schema(parquet_path) == "current" else "retention_time"

        query = f"""
        SELECT
            peptidoform,
            {rt_col} AS retention_time,
            consensus_support,
            posterior_error_probability
        FROM parquet_scan(?)
        {where_clause}
        ORDER BY length(sequence)
        """

        self.query = query
        self.params = [parquet_path] + params
        self.batch_size = validate_batch_size(batch_size)

    def __iter__(self):
        # A fresh connection per iteration: the cursor of a previous pass is already
        # consumed, and DataLoader workers must not share a connection across forks.
        con = duckdb.connect()
        reader = con.execute(self.query, self.params).fetch_record_batch(self.batch_size)

        for batch in reader:
            if batch.num_rows == 0:
                continue
            yield self.process_batch(batch)

    def process_batch(self, batch):
        peptidoform = batch["peptidoform"].to_pylist()
        retention_time = batch["retention_time"].to_pylist()
        rt = np.array(retention_time, dtype=np.float32) / 60.0

        return {
            "peptide": peptidoform,
            "rt": rt
        }


if __name__ == '__main__':
    """Test dataset + sampler + dataloader pipeline."""
    file_path = ["D:/gitrepo/MSnet/MSNetLoader/tests/test_data/PXD014877-Akkermansia_muciniphilia-MSNet.parquet",
                 "D:/gitrepo/MSnet/MSNetLoader/tests/test_data/PXD014877_Clostridium_Bolteae-MSNet.parquet"]
    dataset = RTIterableDataset(file_path)
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False
    )

    batch = next(iter(loader))
    print(batch)
