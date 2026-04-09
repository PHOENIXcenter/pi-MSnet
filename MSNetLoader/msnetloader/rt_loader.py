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


class RTTorchDataset(TorchDataset):

    def __init__(
        self,
        csv_path,
        feature_columns=None,
        label_column="tr",
        output_path="rt.arrow"
    ):
        df = pd.read_csv(csv_path)

        # 计算长度
        df["nAA"] = df["seq"].apply(len)

        # 物理排序（关键！提升 bucket 性能）
        df = df.sort_values("nAA").reset_index(drop=True)

        self.lengths = df["nAA"].values

        self.dataset = self.write_arrow_streaming(
            df,
            output_path=output_path
        )

        self.feature_columns = feature_columns or [
            "seq",
            "modifications",
            "nAA",
        ]

        self.label_column = label_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        features = {
            k: sample[k] for k in self.feature_columns
        }

        label = torch.tensor(
            sample[self.label_column],
            dtype=torch.float32
        )

        return features, label

    @staticmethod
    def write_arrow_streaming(df, output_path, batch_size=10000):

        schema = pa.schema([
            ("seq", pa.string()),
            ("modifications", pa.string()),
            ("nAA", pa.int32()),
            ("tr", pa.float32()),
        ])

        with open(output_path, "wb") as sink:
            with ipc.new_stream(sink, schema) as writer:

                n = len(df)

                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    batch_df = df.iloc[start:end]

                    batch = pa.record_batch({
                        "seq": batch_df["seq"].tolist(),
                        "modifications": batch_df["modifications"].fillna("").tolist(),
                        "nAA": batch_df["nAA"].astype("int32").tolist(),
                        "tr": batch_df["tr"].astype("float32").tolist(),
                    }, schema=schema)

                    writer.write_batch(batch)

        return Dataset.from_file(output_path)


class DeepLCConverter:

    def __init__(self, parquet_path, batch_size=100_000):
        self.parquet_path = parquet_path
        self.batch_size = batch_size
    
    @staticmethod
    def format_modifications_ms2pip_style(mods_raw, seq):

        formatted = []

        for mod in mods_raw:

            if not mod:
                continue

            name = mod.get("name")

            if not name:
                continue

            positions = mod.get("positions", [])

            for p in positions:

                pos = p.get("position")

                if pos is None:
                    continue

                # ------------------------
                # 位置规则
                # ------------------------
                # 如果你 parquet 已经是 1-based，直接用
                # 如果是 0-based，需要 +1

                formatted.append(f"{pos}|{name}")

        return "|".join(formatted)
    
    def convert_parquet_to_rt_format(self):

        parquet_file = pq.ParquetFile(self.parquet_path)

        rows = []

        for batch in parquet_file.iter_batches(batch_size=self.batch_size):

            table = pa.Table.from_batches([batch])

            sequences = table["sequence"]
            modifications = table["modifications"]

            # RT 字段（根据你 parquet 实际字段名改）
            tr_array = table["retention_time"]

            n_rows = table.num_rows

            for i in range(n_rows):

                seq = sequences[i].as_py()

                # --------------------
                # RT
                # --------------------
                tr = tr_array[i].as_py()

                # --------------------
                # modifications
                # --------------------
                mods_formatted = ""

                mods_raw = modifications[i].as_py()

                if mods_raw:
                    mods_formatted = self.format_modifications_ms2pip_style(
                        mods_raw,
                        seq
                    )

                rows.append(
                    (
                        seq,
                        mods_formatted,
                        float(tr) / 60
                    )
                )

        df = pd.DataFrame(
            rows,
            columns=[
                "seq",
                "modifications",
                "tr"
            ],
        )

        return df
