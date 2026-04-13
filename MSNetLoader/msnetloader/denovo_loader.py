import torch
import numpy as np
from torch.utils.data import IterableDataset
import duckdb


class DeNovoIterableDataset(IterableDataset):

    def __init__(self, parquet_path, max_peaks=150, batch_size=32,
                 min_consensus_support=None,
                 max_pep=None):

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
                    exp_mass_to_charge AS precursor_mz,
                    precursor_charge AS charge,
                    mz_array,
                    consensus_support,
                    posterior_error_probability,
                    intensity_array
                FROM parquet_scan(?)
                {where_clause}
                """

        self.cursor = con.execute(query, [parquet_path] + params)

        self.batch_size = batch_size
        self.max_peaks = max_peaks

    def __iter__(self):
        reader = self.cursor.fetch_record_batch(self.batch_size)

        for batch in reader:
            if batch.num_rows == 0:
                continue
            yield self.process_batch(batch)

    # -----------------------------
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

        # -----------------------------
        # per spectrum processing
        # -----------------------------
        for i in range(len(peptidoform)):
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

            spectra_out.append(torch.tensor(spectrum, dtype=torch.float32))
            seq_out.append(peptidoform[i])
            charge_out.append(charges[i])
            precursor_out.append(precursor_mz[i])

        return {
            "spectrum": spectra_out,
            "sequence": seq_out,
            "precursor_mz": torch.tensor(precursor_out, dtype=torch.float32),
            "charge": torch.tensor(charge_out, dtype=torch.long),
        }


if __name__ == '__main__':
    """Test dataset + sampler + dataloader pipeline."""
    file_path = ["D:/gitrepo/MSnet/MSNetLoader/tests/test_data/PXD014877-Akkermansia_muciniphilia-MSNet.parquet",
                 "D:/gitrepo/MSnet/MSNetLoader/tests/test_data/PXD014877_Clostridium_Bolteae-MSNet.parquet"]
    dataset = DeNovoIterableDataset(file_path)
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False
    )

    batch = next(iter(loader))
    print(batch)

