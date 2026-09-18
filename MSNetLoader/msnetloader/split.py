"""Split π-MSNet data into training / validation / test sets.

The split is computed on unique peptide identifiers (``peptidoform`` by
default) using a deterministic hash partition, which guarantees:

* **No leakage** — every PSM sharing a peptide ends up in the same split.
* **Reproducibility** — for a given ``seed`` the same peptide always lands in
  the same split, independent of file order, machine, or Python version.

The partition conditions returned by :func:`split_conditions` are plain DuckDB
SQL ``WHERE`` clauses, so they can be pushed down to the dataset loaders via
``extra_where`` without materializing any data:

>>> from msnetloader import split_conditions
>>> from msnetloader.ms2_loader import MS2TorchDataset
>>> conditions = split_conditions(test_size=0.1, val_size=0.1, seed=42)
>>> train = MS2TorchDataset("PXD021013-MSNet.parquet", extra_where=conditions["train"])
>>> test = MS2TorchDataset("PXD021013-MSNet.parquet", extra_where=conditions["test"])

Use :func:`split_dataset` to write the partitions to standalone parquet files.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Sequence, Union

import duckdb

__all__ = [
    "DEFAULT_NUM_BUCKETS",
    "DEFAULT_SPLIT_KEY",
    "split_conditions",
    "split_dataset",
    "split_files",
    "split_peptides",
]

DEFAULT_SPLIT_KEY = "peptidoform"
DEFAULT_NUM_BUCKETS = 10000

PathLike = Union[str, os.PathLike]


def _validate_fractions(test_size: float, val_size: float) -> None:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size!r}")
    if not 0.0 <= val_size < 1.0:
        raise ValueError(f"val_size must be in [0, 1), got {val_size!r}")
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1, got {test_size + val_size!r}")


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 63:
        raise ValueError(f"seed must be an integer in [0, 2^63), got {seed!r}")


def _bucket_expr(key: str, seed: int, num_buckets: int) -> str:
    """DuckDB expression mapping ``key`` to a stable bucket in [0, num_buckets).

    ``hash()`` returns an unsigned 64-bit integer and ``xor`` with the seed
    keeps the result unsigned; the double modulo normalises the remainder.
    """
    return f"((xor(hash({key}), {seed}) % {num_buckets}) + {num_buckets}) % {num_buckets}"


def split_conditions(
    test_size: float = 0.1,
    val_size: float = 0.0,
    seed: int = 42,
    key: str = DEFAULT_SPLIT_KEY,
    num_buckets: int = DEFAULT_NUM_BUCKETS,
) -> dict[str, str]:
    """Return DuckDB SQL ``WHERE`` conditions for the train/val/test splits.

    The split is a hash partition of the distinct values of *key* into
    ``num_buckets`` buckets: the first ``round(test_size * num_buckets)``
    buckets form the test split, the next ``round(val_size * num_buckets)``
    form the validation split, and the remainder the training split.

    Parameters
    ----------
    test_size:
        Approximate fraction of *peptides* assigned to the test split.
    val_size:
        Approximate fraction of *peptides* assigned to the validation split
        (``0`` disables it).
    seed:
        Random seed; changing it reshuffles which peptides land where.
    key:
        Column used as the split unit. Defaults to ``peptidoform`` so that all
        PSMs of one peptide stay together; use ``sequence`` to ignore
        modification state.
    num_buckets:
        Granularity of the hash partition; higher values give split sizes
        closer to the requested fractions.

    Returns
    -------
    dict[str, str]
        Mapping ``{"train": ..., "val": ..., "test": ...}`` to SQL conditions
        such as ``((xor(hash(peptidoform), 42) % 10000) + 10000) % 10000 >= 2000``.
    """
    _validate_fractions(test_size, val_size)
    _validate_seed(seed)
    if num_buckets < 2:
        raise ValueError(f"num_buckets must be >= 2, got {num_buckets!r}")

    bucket = _bucket_expr(key, seed, num_buckets)
    test_buckets = max(1, round(test_size * num_buckets))
    val_buckets = max(1, round(val_size * num_buckets)) if val_size > 0 else 0
    val_start = test_buckets

    return {
        "test": f"{bucket} < {test_buckets}",
        "val": f"({bucket} >= {val_start} AND {bucket} < {val_start + val_buckets})",
        "train": f"{bucket} >= {val_start + val_buckets}",
    }


def _normalise_paths(parquet_paths: Union[PathLike, Sequence[PathLike]]) -> list[Path]:
    if isinstance(parquet_paths, (str, os.PathLike)):
        parquet_paths = [parquet_paths]
    return [Path(p) for p in parquet_paths]


def _sql_string(path: Union[Path, str]) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _source_expr(parquet_paths: Sequence[Path]) -> str:
    quoted = ", ".join(f"'{_sql_string(p)}'" for p in parquet_paths)
    return f"parquet_scan([{quoted}])"


def _open_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def split_peptides(
    parquet_paths: Union[PathLike, Sequence[PathLike]],
    test_size: float = 0.1,
    val_size: float = 0.0,
    seed: int = 42,
    key: str = DEFAULT_SPLIT_KEY,
    num_buckets: int = DEFAULT_NUM_BUCKETS,
) -> dict[str, list[str]]:
    """Return the distinct *key* values (e.g. peptidoforms) of each split.

    Reads only the *key* column of the parquet file(s); use this when the
    peptide lists themselves are needed (e.g. to filter downstream data).
    """
    paths = _normalise_paths(parquet_paths)
    conditions = split_conditions(test_size, val_size, seed, key, num_buckets)
    source = _source_expr(paths)

    con = _open_connection()
    result: dict[str, list[str]] = {}
    for split, condition in conditions.items():
        rows = con.execute(f"SELECT DISTINCT {key} FROM {source} WHERE {condition}").fetchall()
        result[split] = [row[0] for row in rows]
    return result


def split_dataset(
    parquet_paths: Union[PathLike, Sequence[PathLike]],
    output_dir: Union[PathLike],
    test_size: float = 0.1,
    val_size: float = 0.0,
    seed: int = 42,
    key: str = DEFAULT_SPLIT_KEY,
    num_buckets: int = DEFAULT_NUM_BUCKETS,
    prefix: Optional[str] = None,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Write train/val/test parquet files split at the peptide level.

    Every row of the input file(s) is written to exactly one output file based
    on the hash partition of *key* (see :func:`split_conditions`), so the
    outputs can be consumed by the same loaders as the inputs.

    Parameters
    ----------
    parquet_paths:
        One or more ``*-MSNet.parquet`` files.
    output_dir:
        Directory receiving ``<prefix>train.parquet``, ``<prefix>val.parquet``
        and ``<prefix>test.parquet``.
    prefix:
        Output file name prefix. Defaults to ``<accession>_`` when a single
        input file is given, otherwise to ``"split_"``.
    overwrite:
        Allow overwriting existing output files. When ``False`` and an output
        file already exists, :class:`FileExistsError` is raised.

    Returns
    -------
    dict[str, list[Path]]
        Mapping of split name to written output path(s).
    """
    _validate_fractions(test_size, val_size)
    paths = _normalise_paths(parquet_paths)
    if not paths:
        raise ValueError("parquet_paths must not be empty")

    conditions = split_conditions(test_size, val_size, seed, key, num_buckets)
    source = _source_expr(paths)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        if len(paths) == 1:
            stem = paths[0].name.split("-")[0]
            prefix = f"{stem}_"
        else:
            prefix = "split_"

    splits = ["train", "val", "test"] if val_size > 0 else ["train", "test"]
    con = _open_connection()
    results: dict[str, list[Path]] = {}
    for split in splits:
        out_path = output_dir / f"{prefix}{split}.parquet"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} already exists; pass overwrite=True to replace it")
        target = _sql_string(out_path)
        copy_options = "(FORMAT PARQUET, OVERWRITE_OR_IGNORE 1)" if overwrite else "(FORMAT PARQUET)"
        con.execute(f"COPY (SELECT * FROM {source} WHERE {conditions[split]}) TO '{target}' {copy_options}")
        results[split] = [out_path]
    return results


def split_files(
    file_paths: Sequence[PathLike],
    test_size: float = 0.1,
    val_size: float = 0.0,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Split a list of parquet files into train/val/test lists.

    A coarse, file-granularity split for multi-project training, where each
    file (usually one project/species) goes to a single split. Prefer
    :func:`split_conditions` or :func:`split_dataset` when peptides should be
    split across a single collection of files.
    """
    _validate_fractions(test_size, val_size)
    paths = [Path(p) for p in file_paths]

    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(test_size * len(shuffled))) if shuffled else 0
    n_val = max(1, round(val_size * len(shuffled))) if val_size > 0 and shuffled else 0

    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


if __name__ == "__main__":
    print(split_conditions(test_size=0.1, val_size=0.1, seed=42))
