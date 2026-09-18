"""msnetloader: a Python API for seamless integration of π-MSNet into AI workflows."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

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

try:
    __version__ = _package_version("msnetloader")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

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
