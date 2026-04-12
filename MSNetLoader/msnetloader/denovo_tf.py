import tensorflow as tf
import numpy as np
import duckdb


class DeNovoTFDataset:

    def __init__(self, parquet_path, max_peaks=150, batch_size=32,
                 min_consensus_support=None,
                 max_pep=None):

        con = duckdb.connect()

        query = f"""
            SELECT
                peptidoform,
                exp_mass_to_charge AS precursor_mz,
                precursor_charge AS charge,
                mz_array,
                consensus_support,
                posterior_error_probability,
                intensity_array
            FROM parquet_scan('{parquet_path}')
        """

        self.arrow_table = con.execute(query).to_arrow_table()

        self.batch_size = batch_size
        self.max_peaks = max_peaks
        self.min_consensus_support = min_consensus_support
        self.max_pep = max_pep

    # =========================================================
    # Generator (核心)
    # =========================================================
    def generator(self):

        for batch in self.arrow_table.to_batches(max_chunksize=self.batch_size):
            result = self.process_batch(batch)

            for i in range(len(result["sequence"])):
                yield (
                    result["spectrum"][i],
                    result["sequence"][i],
                    result["precursor_mz"][i],
                    result["charge"][i],
                )

    # =========================================================
    # TF Dataset接口
    # =========================================================
    def get_dataset(self):

        output_signature = (
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),  # spectrum
            tf.TensorSpec(shape=(), dtype=tf.string),          # sequence
            tf.TensorSpec(shape=(), dtype=tf.float32),         # precursor_mz
            tf.TensorSpec(shape=(), dtype=tf.int32),           # charge
        )

        ds = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=output_signature
        )

        return ds

    # =========================================================
    # batch处理（基本不变）
    # =========================================================
    def process_batch(self, batch):

        peptidoform = batch["peptidoform"].to_pylist()
        charges = batch["charge"].to_pylist()
        mz_list = batch["mz_array"].to_pylist()
        int_list = batch["intensity_array"].to_pylist()
        precursor_mz = batch["precursor_mz"].to_pylist()
        consensus_supports = batch["consensus_support"].to_pylist()
        peps = batch["posterior_error_probability"].to_pylist()

        spectra_out = []
        seq_out = []
        charge_out = []
        precursor_out = []

        for i in range(len(peptidoform)):

            if not self.filter_by_consensus_support(consensus_supports[i]):
                continue

            if not self.filter_by_pep(peps[i]):
                continue

            mz = np.asarray(mz_list[i], dtype=np.float32)
            intensity = np.asarray(int_list[i], dtype=np.float32)

            if len(mz) == 0:
                continue

            # -------------------------
            # top-k peaks
            # -------------------------
            if len(mz) > self.max_peaks:
                idx = np.argsort(intensity)[-self.max_peaks:]
                mz = mz[idx]
                intensity = intensity[idx]

            # -------------------------
            # normalize
            # -------------------------
            max_int = intensity.max() if len(intensity) > 0 else 1.0
            intensity = intensity / max_int

            spectrum = np.stack([mz, intensity], axis=1)

            spectra_out.append(spectrum.astype(np.float32))
            seq_out.append(peptidoform[i].encode("utf-8"))  # TF需要bytes
            charge_out.append(np.int32(charges[i]))
            precursor_out.append(np.float32(precursor_mz[i]))

        return {
            "spectrum": spectra_out,
            "sequence": seq_out,
            "precursor_mz": precursor_out,
            "charge": charge_out,
        }

    # =========================================================
    def filter_by_consensus_support(self, support):
        if self.min_consensus_support is None:
            return True
        if support is None:
            return False
        return support >= self.min_consensus_support

    # =========================================================
    def filter_by_pep(self, pep):
        if self.max_pep is None:
            return True
        if pep is None:
            return False
        return pep <= self.max_pep