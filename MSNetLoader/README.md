# MSNetLoader
A Python API for seamless integration of π-MSNet into AI workflows

## Install

```bash
pip install -e .
```

## Download a project dataset

Download all (or selected) files of a project accession from the quantms
repository. The default source is the official EBI FTP mirror; the
browse.quantms.org listing is also supported, and full dataset URLs are
accepted as-is:

```python
from msnetloader import download_dataset, list_remote_files

# see what a project ships
list_remote_files("PXD021013")
# ['PXD021013-MSNet.parquet', 'PXD021013.dataset.parquet', ...]

# download only the small metadata files
downloaded = download_dataset(
    "PXD021013",
    data_dir="data/PXD021013",
    files=["dataset", "run", "sample"],
)

# download everything, including the (large) main PSM parquet
download_dataset("PXD021013", data_dir="data/PXD021013", files="all")

# or from the browse.quantms.org page of a specific dataset version
download_dataset(
    "https://browse.quantms.org/quantms/datasets/PXD021013/1ed4852abc60/",
    data_dir="data/PXD021013",
    files=["msnet"],
)
```

Interrupted downloads are resumed automatically from the `*.part` file;
pass `force=True` to re-download existing files.

## Split data into train / validation / test sets

Splits are computed as a deterministic hash partition of the peptide
identifier (`peptidoform`), so all PSMs of a peptide stay in the same split
(no data leakage) and the same peptide always lands in the same split for a
given seed.

Push the split conditions down to the loaders without copying any data:

```python
from msnetloader import split_conditions
from msnetloader.ms2_loader import MS2TorchDataset

conditions = split_conditions(test_size=0.1, val_size=0.1, seed=42)

train_set = MS2TorchDataset("PXD021013-MSNet.parquet", extra_where=conditions["train"])
val_set = MS2TorchDataset("PXD021013-MSNet.parquet", extra_where=conditions["val"])
test_set = MS2TorchDataset("PXD021013-MSNet.parquet", extra_where=conditions["test"])
```

Or materialize standalone train/test parquet files:

```python
from msnetloader import split_dataset, split_files, split_peptides

# write <prefix>train.parquet / <prefix>test.parquet
split_dataset("PXD021013-MSNet.parquet", output_dir="splits", test_size=0.1, seed=42)

# get the peptide lists per split
peptides = split_peptides("PXD021013-MSNet.parquet", test_size=0.1, seed=42)

# coarse file-level split across multiple projects
split_files(["PXD021013-MSNet.parquet", "PXD014877-MSNet.parquet"], test_size=0.2, seed=42)
```

## Load datasets

```python
from msnetloader.ms2_loader import MS2TorchDataset
from torch.utils.data import DataLoader

dataset = MS2TorchDataset("test_data/PXD014877-Akkermansia_muciniphilia-MSNet.parquet", ion_types=("b", "y"))
dataloader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=False)

for batch in dataloader:
    ...
```

## Run tests

```bash
pytest tests
# opt-in live network test against the real quantms mirror:
MSNETLOADER_LIVE_TESTS=1 pytest tests/test_download.py
```
