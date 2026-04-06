import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from datasets import Dataset
import pyarrow.ipc as ipc
from torch.utils.data import Sampler
import math
import random
from datasets import Dataset as ArrowDataset


def denovo_collate_fn(batch):

    spectra = [b["spectrum"] for b in batch]
    tokens = [b["sequence"] for b in batch]

    precursor_mz = torch.stack([b["precursor_mz"] for b in batch])
    charge = torch.stack([b["charge"] for b in batch])

    # -----------------
    # spectrum padding
    # -----------------

    max_peaks = max(s.shape[0] for s in spectra)

    padded_spec = []

    for s in spectra:

        pad_len = max_peaks - s.shape[0]

        if pad_len > 0:
            pad = torch.zeros(pad_len, 2)
            s = torch.cat([s, pad], dim=0)

        padded_spec.append(s)

    spectra = torch.stack(padded_spec)

    return {
        "spectra": spectra,
        "precursor_mz": precursor_mz,
        "charge": charge,
        "tokens": tokens
    }


class PeptideTokenizer:

    def __init__(self):

        aas = list("ACDEFGHIKLMNPQRSTVWY")

        self.pad = "<pad>"
        self.bos = "<bos>"
        self.eos = "<eos>"

        self.vocab = [self.pad, self.bos, self.eos] + aas

        self.stoi = {a:i for i,a in enumerate(self.vocab)}
        self.itos = {i:a for i,a in enumerate(self.vocab)}

        self.pad_id = self.stoi[self.pad]
        self.bos_id = self.stoi[self.bos]
        self.eos_id = self.stoi[self.eos]

    def encode(self, seq):

        tokens = [self.bos_id]

        for a in seq:
            tokens.append(self.stoi.get(a, self.pad_id))

        tokens.append(self.eos_id)

        return tokens


class DeNovoDataset(TorchDataset):

    def __init__(self, data_path, max_peaks=150):

        self.max_peaks = max_peaks

        if data_path.endswith(".arrow"):

            reader = ipc.open_file(data_path)
            self.table = reader.read_all()
            self.backend = "arrow"
            self.length = self.table.num_rows

        elif data_path.endswith(".parquet"):

            self.parquet = pq.ParquetFile(data_path)
            self.backend = "parquet"
            self.length = self.parquet.metadata.num_rows

            # row group size
            self.row_group_size = self.parquet.metadata.row_group(0).num_rows

        else:
            raise ValueError("Only support .arrow or .parquet")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.backend == "arrow":

            row = self.table.slice(idx, 1)

            mz = row["mz"][0].as_py()
            intensity = row["intensity"][0].as_py()
            sequence = row["sequence"][0].as_py()
            charge = row["precursor_charge"][0].as_py()
            precursor_mz = row["precursor_mz"][0].as_py()

        else:

            group = idx // self.row_group_size
            offset = idx % self.row_group_size

            table = self.parquet.read_row_group(group)

            mz = table["mz_array"][offset].as_py()
            intensity = table["intensity_array"][offset].as_py()
            sequence = table["sequence"][offset].as_py()
            charge = table["precursor_charge"][offset].as_py()
            precursor_mz = table["exp_mass_to_charge"][offset].as_py()

        # numpy
        mz = np.asarray(mz)
        intensity = np.asarray(intensity)

        # top peaks
        if len(mz) > self.max_peaks:

            idxs = np.argsort(intensity)[-self.max_peaks:]
            mz = mz[idxs]
            intensity = intensity[idxs]

        # normalize
        if intensity.max() > 0:
            intensity = intensity / intensity.max()

        spectrum = np.stack([mz, intensity], axis=1)

        spectrum = torch.tensor(spectrum, dtype=torch.float32)

        # tokens = torch.tensor(sequence, dtype=torch.long)

        precursor_mz = torch.tensor(precursor_mz, dtype=torch.float32)
        charge = torch.tensor(charge, dtype=torch.long)

        return {
            "spectrum": spectrum,
            "sequence": sequence,
            "precursor_mz": precursor_mz,
            "charge": charge
        }


class DeNovoArrowConverter:

    def __init__(self, parquet_path, batch_size=100000):
        self.parquet_path = parquet_path
        self.batch_size = batch_size

    def convert(self, output_path):

        parquet_file = pq.ParquetFile(self.parquet_path)
        schema = pa.schema([
            ("sequence", pa.string()),
            ("precursor_charge", pa.int32()),
            ("precursor_mz", pa.float32()),
            ("mz", pa.list_(pa.float32())),
            ("intensity", pa.list_(pa.float32())),
        ])

        with ipc.new_file(output_path, schema) as writer:

            for batch in parquet_file.iter_batches(batch_size=self.batch_size):

                table = pa.Table.from_batches([batch])
                print(table)
                new_table = pa.table({
                    "sequence": table["sequence"],
                    "precursor_charge": table["precursor_charge"],
                    "precursor_mz": table["exp_mass_to_charge"],
                    "mz": table["mz_array"],
                    "intensity": table["intensity_array"],
                }, schema=schema)

                writer.write_table(new_table)

        print("Arrow file written to:", output_path)


if __name__ == "__main__":
    tokenizer = PeptideTokenizer()
    dataset = DeNovoDataset("/mnt/daicx/pvc-afbfaa68-aa52-416c-b273-64fb016fd745/msnet_paper/released_data_v3/PXD012636_Danio_rerio-MSNet.parquet")
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=denovo_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    for k, v in next(iter(loader)).items():
        print("Key:", k)
        print("Type:", type(v))
        print("Shape:", getattr(v, "shape", None))
        print(v)