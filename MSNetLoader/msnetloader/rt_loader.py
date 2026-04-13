from torch.utils.data import IterableDataset
import duckdb
import numpy as np


class RTIterableDataset(IterableDataset):

    def __init__(self, parquet_path, batch_size=32,
                 min_consensus_support=None,
                 max_pep=None
                 ):
        con = duckdb.connect()
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

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        SELECT
            peptidoform,
            retention_time,
            consensus_support,
            posterior_error_probability
        FROM parquet_scan(?)
        {where_clause}
        ORDER BY length(sequence)
        """

        self.cursor = con.execute(query, [parquet_path] + params)
        self.batch_size = batch_size

    def __iter__(self):
        reader = self.cursor.fetch_record_batch(self.batch_size)

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