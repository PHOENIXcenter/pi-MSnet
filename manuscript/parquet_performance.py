import glob
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetAppender:
    def __init__(self, path: str):
        self.path = path
        self.writer = None
        self.schema = [
            pa.field('sequence', pa.string()),
            pa.field('charge', pa.int8()),
            pa.field('USI', pa.string()),
            pa.field('peptidoform', pa.string()),
            pa.field('exp_mass_to_charge', pa.float32()),
            pa.field('global_qvalue', pa.float32()),
            pa.field('retention_time', pa.float32()),
            pa.field('consensus_support', pa.float32()),
            pa.field('mz_array', pa.list_(pa.float32())),
            pa.field('intensity_array', pa.list_(pa.float32())),
            pa.field('ions_matched', pa.list_(pa.string()))
        ]

    def append(self, df: pd.DataFrame):
        table = pa.Table.from_pandas(df, schema=pa.schema(self.schema))
        if self.writer is None:
            if os.path.exists(self.path):
                existing_table = pq.read_table(self.path)
                self.writer = pq.ParquetWriter(self.path, existing_table.schema, compression='gzip')
                self.writer.write_table(existing_table)
            else:
                self.writer = pq.ParquetWriter(self.path, table.schema, compression='gzip')
        self.writer.write_table(table)

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None


def plot_file_size():
    apppender = ParquetAppender("./test.parquet")
    files = glob.glob(
        "/mnt/daicx/pvc-afbfaa68-aa52-416c-b273-64fb016fd745/Cohort_E480_DDAQC/huiyan_msnet/ionmatched/*_RT.parquet")[
            :300]
    output_hdf = "./test.h5"
    with pd.HDFStore(output_hdf, mode='w') as store:
        for i, f in enumerate(files):
            data = pd.read_parquet(f)
            apppender.append(data)
            basename = f.replace(
                "/mnt/daicx/pvc-afbfaa68-aa52-416c-b273-64fb016fd745/Cohort_E480_DDAQC/huiyan_msnet/ionmatched", ".")
            data.to_csv("./test.csv", index=False, mode="a", header=(i == 0))
            data['mz_array'] = data['mz_array'].apply(lambda x: str(list(x)))
            data['intensity_array'] = data['intensity_array'].apply(lambda x: str(list(x)))
            data['ions_matched'] = data['ions_matched'].apply(lambda x: str(list(x)))
            data['consensus_support'] = data['consensus_support'].astype(np.float32)  # or float32, whichever is needed
            store.append("data", data, format="table", data_columns=True, min_itemsize={"mz_array": 20000,
                                                                                        "intensity_array": 20000,
                                                                                        "ions_matched": 20000})

            # size_bytes = os.path.getsize(i)
            # size_mb = size_bytes / (1024 * 1024)
            # print(size_mb)

            # data.to_csv(basename.replace(".parquet", ".csv"), index=False)
            # data.to_hdf(basename.replace(".parquet", ".h5"), key="data", mode="w")
            data.to_pickle(basename.replace(".parquet", ".pkl"))
            # print(data.columns)
            # print(data.dtypes)

            # size_bytes = os.path.getsize(basename.replace(".parquet", ".csv"))
            # size_mb = size_bytes / (1024 * 1024)
            # print(size_mb)
            #
            # size_bytes = os.path.getsize(basename.replace(".parquet", ".h5"))
            # size_mb = size_bytes / (1024 * 1024)
            # print(size_mb)
            #
            # size_bytes = os.path.getsize(basename.replace(".parquet", ".pkl"))
            # size_mb = size_bytes / (1024 * 1024)
            # print(size_mb)

        apppender.close()
    # i = "/mnt/daicx/pvc-afbfaa68-aa52-416c-b273-64fb016fd745/pOpenDeep/data/IPX0000937001/IPX0000937001_psm.parquet"
    # size_bytes = os.path.getsize(i)
    # size_mb = size_bytes / (1024 * 1024)
    # print(size_mb)
    # pf = pq.ParquetFile(i)
    # print(pf.metadata.num_rows)  # 24924726
    # # data = pd.read_parquet(i)
    # basename = i.replace("/mnt/daicx/pvc-afbfaa68-aa52-416c-b273-64fb016fd745/pOpenDeep/data/IPX0000937001", ".")
    #
    # # data.to_csv(basename.replace(".parquet", ".csv"), index=False)
    # # data.to_hdf(basename.replace(".parquet", ".h5"), key="data", mode="w")
    # # data.to_pickle(basename.replace(".parquet", ".pkl"))
    #
    # first_chunk = True
    # compress_hdf=True
    #
    # # iterate over row groups to avoid loading entire file
    # for rg in range(pf.num_row_groups):
    #     table = pf.read_row_group(rg)          # pyarrow.Table for this row-group
    #     df = table.to_pandas()                 # convert small chunk to pandas
    #     print(df.columns)
    #     print(df.dtypes)
    #     # write CSV (append)
    #     if first_chunk:
    #         df.to_csv(basename.replace(".parquet", ".csv"), index=False, mode="w")
    #     else:
    #         df.to_csv(basename.replace(".parquet", ".csv"), index=False, mode="a", header=False)
    #
    #     # write HDF (appendable format)
    #     # if hdf_path is not None:
    #     # use format='table' to allow append
    #     df.to_hdf(basename.replace(".parquet", ".h5"), key='data', mode="a" if not first_chunk else "w",
    #               format="table", complevel=9, complib="blosc" if compress_hdf else None)
    #
    #     first_chunk = False
    #
    # print("Done:")
    #
    # print("Row groups:", pf.num_row_groups)
    # size_bytes = os.path.getsize(basename.replace(".parquet", ".csv"))
    # size_mb = size_bytes / (1024 * 1024)
    # print(size_mb)
    #
    # size_bytes = os.path.getsize(basename.replace(".parquet", ".h5"))
    # size_mb = size_bytes / (1024 * 1024)
    # print(size_mb)
    #
    # size_bytes = os.path.getsize(basename.replace(".parquet", ".pkl"))
    # size_mb = size_bytes / (1024 * 1024)
    # print(size_mb)
    #
    # compressions = set()
    # for i in range(pf.num_row_groups):
    #     for j in range(pf.metadata.row_group(i).num_columns):
    #         compressions.add(pf.metadata.row_group(i).column(j).compression)
    # print(compressions)


def psm_number_and_file_size():
    input_file = "./test.parquet"
    # pf = pq.ParquetFile("./test.parquet")
    # print(pf.metadata.num_rows)  # 24924726

    output_dir = "./"

    split_sizes = [10_000, 100_000, 500_000, 1_000_000, 1888575]

    # Read the entire Parquet table (can also stream by row_group to save memory)
    table = pq.read_table(input_file)
    total_rows = table.num_rows

    print(f"Total rows: {total_rows}")

    # Convert to Pandas DataFrame for easy slicing
    df = table.to_pandas()

    # Generate progressively cumulative output files
    for size in split_sizes:
        end = min(size, total_rows)
        df_part = df.iloc[:end]

        # File name (e.g. split_10k.parquet / split_10k.csv / split_10k.h5)
        suffix = f"{size // 1000}k"
        parquet_out = os.path.join(output_dir, f"split_{suffix}.parquet")
        csv_out = os.path.join(output_dir, f"split_{suffix}.csv")
        h5_out = os.path.join(output_dir, f"split_{suffix}.h5")

        # Write three formats
        # df_part.to_parquet(parquet_out, index=False)
        apppender = ParquetAppender(parquet_out)
        apppender.append(df_part)
        apppender.close()
        df_part.to_csv(csv_out, index=False)

        # ⚠️ HDF5 requires basic field types (avoid object columns)
        df_simple = df_part.copy()
        for col in df_simple.columns:
            if df_simple[col].dtype == "object":
                df_simple[col] = df_simple[col].astype(str)

        df_simple.to_hdf(h5_out, key="data", mode="w", format="table")

        print(f"✅ Generated {suffix}: {len(df_part)} rows ->")
        print(f"   {parquet_out}")
        print(f"   {csv_out}")
        print(f"   {h5_out}")


def plot_scatter():
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams.update({
        "font.size": 16,  # default font
        "axes.titlesize": 18,  # title font
        "axes.labelsize": 16,  # axis label font
        "xtick.labelsize": 14,  # X axis tick font
        "ytick.labelsize": 14,  # Y axis tick font
        "legend.fontsize": 14,  # legend font
    })

    # Data
    psm_number = [10_000, 100_000, 500_000, 1_000_000, 1_888_575]
    parquet_file_size = ["20M", "200M", "1.2G", "2.3G", "4.4G"]
    csv_file_size = ["59M", "646M", "3.8G", "7.4G", "15G"]
    hdf5_file_size = ["680M", "8.5G", "42.6G", "74G", "113G"]

    # Normalize units to MB
    def parse_size(x):
        x = str(x).upper().strip()
        if "G" in x:
            return float(x.replace("G", "")) * 1024
        elif "M" in x:
            return float(x.replace("M", ""))
        else:
            return float(x)

    parquet_MB = [parse_size(x) for x in parquet_file_size]
    csv_MB = [parse_size(x) for x in csv_file_size]
    hdf5_MB = [parse_size(x) for x in hdf5_file_size]

    # Build DataFrame (convenient for plotting)
    df = pd.DataFrame({
        "PSM number": psm_number,
        "Parquet (MB)": parquet_MB,
        "CSV (MB)": csv_MB,
        "HDF5 (MB)": hdf5_MB
    })

    # Plot
    plt.figure(figsize=(8, 6), dpi=500)
    plt.plot(df["PSM number"], df["Parquet (MB)"], marker="o", label="Parquet", linewidth=2, color='#FF7F0E')
    plt.plot(df["PSM number"], df["HDF5 (MB)"], marker="^", label="HDF5", linewidth=2, color='#2ca02c')
    plt.plot(df["PSM number"], df["CSV (MB)"], marker="s", label="CSV", linewidth=2, color='#1f77b4')

    for x, y, label in zip(psm_number, parquet_MB, parquet_file_size):
        plt.text(x, y * 1.1, label, fontsize=9, ha='center')
    for x, y, label in zip(psm_number, csv_MB, csv_file_size):
        plt.text(x, y * 1.1, label, fontsize=9, ha='center')
    for x, y, label in zip(psm_number, hdf5_MB, hdf5_file_size):
        plt.text(x, y * 1.1, label, fontsize=9, ha='center')

    # Axes and styling
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("PSM number", fontsize=14)
    plt.ylabel("File size (MB, log scale)", fontsize=14)
    plt.title("File size vs. PSM number", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("file_size.png")


def plot_io_time():
    import os
    import time
    import pandas as pd
    import matplotlib.pyplot as plt

    # Define files and their types
    files = [
        ("split_10k.csv", "CSV"),
        ("split_10k.parquet", "Parquet"),
        ("split_10k.h5", "HDF5"),
        ("split_100k.csv", "CSV"),
        ("split_100k.parquet", "Parquet"),
        ("split_100k.h5", "HDF5"),
        ("split_500k.csv", "CSV"),
        ("split_500k.parquet", "Parquet"),
        ("split_500k.h5", "HDF5"),
        ("split_1000k.csv", "CSV"),
        ("split_1000k.parquet", "Parquet"),
        ("split_1000k.h5", "HDF5"),
        ("split_1888k.csv", "CSV"),
        ("split_1888k.parquet", "Parquet"),
        # ("split_1888k.h5", "HDF5"),
    ]
    
    results = []
    
    for f, ftype in files:
        if not os.path.exists(f):
            continue
    
        size_MB = os.path.getsize(f) / (1024 * 1024)
    
        print(f"Reading {f} ({ftype}, {size_MB:.2f} MB)...", flush=True)
    
        start = time.time()
        try:
            if ftype == "CSV":
                pd.read_csv(f)
            elif ftype == "Parquet":
                pd.read_parquet(f)
            elif ftype == "HDF5":
                pd.read_hdf(f, key="data")
            elapsed = time.time() - start
        except Exception as e:
            print(f"❌ Error reading {f}: {e}")
            elapsed = None
    
        results.append((f, ftype, size_MB, elapsed))
    
    # Convert to DataFrame
    df = pd.DataFrame(results, columns=["filename", "type", "size_MB", "read_time_s"])
    print(df)
    df.to_csv("read_benchmark.csv", index=False)

    plt.rcParams.update({
        "font.size": 16,  # default font
        "axes.titlesize": 18,  # title font
        "axes.labelsize": 16,  # axis label font
        "xtick.labelsize": 14,  # X axis tick font
        "ytick.labelsize": 14,  # Y axis tick font
        "legend.fontsize": 14,  # legend font
    })

    df = pd.read_csv("read_benchmark.csv")
    df = df._append({"filename": "split_1888k.h5", "type": "HDF5", "size_MB": 50000, "read_time_s": 520}, ignore_index=True)

    df["PSM"] = [10_000, 10_000, 10_000,
                        100_000, 100_000, 100_000,
                        500_000, 500_000, 500_000,
                        1_000_000, 1_000_000, 1_000_000,
                        1_888_575, 1_888_575, 1_888_575]

    plt.figure(figsize=(8, 6), dpi=500)
    for file_type in df["type"].unique():
        subset = df[df["type"] == file_type]
        plt.plot(subset["PSM"], subset["read_time_s"], marker="o", label=file_type)
        for _, row in subset.iterrows():
            plt.text(row["PSM"], row["read_time_s"] * 1.02, f'{row["read_time_s"]:.1f}s', ha='center', fontsize=8)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("PSM number (log scale)")
    plt.ylabel("Read time (seconds, log scale)")
    plt.title("File Read Time vs PSM Number for Different File Formats")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("file_io.png")


if __name__ == '__main__':
    plot_file_size()
    psm_number_and_file_size()
    plot_io_time()
    plot_scatter()
