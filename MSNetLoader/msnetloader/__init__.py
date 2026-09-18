"""msnetloader: a Python API for seamless integration of π-MSNet into AI workflows."""

from msnetloader.download import (
    download_by_metadata,
    download_dataset,
    download_datasets,
    fetch_collection_metadata,
    list_remote_files,
    search_datasets,
)
from msnetloader.split import (
    split_conditions,
    split_dataset,
    split_files,
    split_peptides,
)

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "download_by_metadata",
    "download_dataset",
    "download_datasets",
    "fetch_collection_metadata",
    "list_remote_files",
    "search_datasets",
    "split_conditions",
    "split_dataset",
    "split_files",
    "split_peptides",
]
