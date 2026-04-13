import tensorflow as tf
import numpy as np
import duckdb


class RTTFDataset:

    def __init__(
        self,
        parquet_path,
        batch_size=100_000,
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

    # =========================================================
    # generator
    # =========================================================
    def generator(self):
        reader = self.cursor.fetch_record_batch(self.batch_size)
        for batch in reader:
            if batch.num_rows == 0:
                continue
            yield self.process_batch(batch)

    # =========================================================
    def get_dataset(self):
        output_signature = {
            "peptide": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "rt": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        }

        return tf.data.Dataset.from_generator(
            self.generator,
            output_signature=output_signature
        )

    # =========================================================
    def process_batch(self, batch):
        peptidoform = batch["peptidoform"].to_pylist()
        retention_time = batch["retention_time"].to_pylist()
        consensus_support = batch["consensus_support"].to_pylist()
        pep = batch["posterior_error_probability"].to_pylist()

        # ✅ 转 numpy
        peptidoform = np.array(peptidoform, dtype=np.string_)
        retention_time = np.array(retention_time, dtype=np.float32)
        consensus_support = np.array(consensus_support)
        pep = np.array(pep)

        # =====================================================
        # ✅ 应用 filter（重点）
        # =====================================================
        mask = np.ones(len(peptidoform), dtype=bool)

        if self.min_consensus_support is not None:
            mask &= (consensus_support >= self.min_consensus_support)

        if self.max_pep is not None:
            mask &= (pep <= self.max_pep)

        peptidoform = peptidoform[mask]
        retention_time = retention_time[mask]

        # =====================================================
        return {
            "peptide": peptidoform,
            "rt": retention_time / 60.0  # s → m
        }