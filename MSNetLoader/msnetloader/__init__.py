"""msnetloader: a Python API for seamless integration of π-MSNet into AI workflows."""

from msnetloader.download import download_dataset, list_remote_files
from msnetloader.split import (
    split_conditions,
    split_dataset,
    split_files,
    split_peptides,
)

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "download_dataset",
    "list_remote_files",
    "split_conditions",
    "split_dataset",
    "split_files",
    "split_peptides",
]
