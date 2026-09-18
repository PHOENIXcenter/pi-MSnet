"""Tests for train/val/test split utilities."""

import logging
from pathlib import Path

import duckdb
import pytest
from torch.utils.data import DataLoader

from msnetloader import (
    split_conditions,
    split_dataset,
    split_files,
    split_peptides,
)
from msnetloader.ms2_loader import MS2TorchDataset

TESTS_DIR = Path(__file__).parent
TEST_PARQUETS = [
    str(TESTS_DIR / "test_data/PXD014877-Akkermansia_muciniphilia-MSNet.parquet"),
    str(TESTS_DIR / "test_data/PXD014877_Clostridium_Bolteae-MSNet.parquet"),
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _source(paths=None):
    quoted = ", ".join(f"'{p}'" for p in (paths or TEST_PARQUETS))
    return f"parquet_scan([{quoted}])"


def _count_rows(condition, paths=None):
    con = duckdb.connect()
    return con.execute(f"SELECT count(*) FROM {_source(paths)} WHERE {condition}").fetchone()[0]


def _distinct_keys(condition, key="peptidoform", paths=None):
    con = duckdb.connect()
    rows = con.execute(f"SELECT DISTINCT {key} FROM {_source(paths)} WHERE {condition}").fetchall()
    return {row[0] for row in rows}


def test_split_conditions_partition_all_rows():
    """Every row must be assigned to exactly one of train/val/test."""
    conditions = split_conditions(test_size=0.1, val_size=0.1, seed=42)

    con = duckdb.connect()
    total = con.execute(f"SELECT count(*) FROM {_source()}").fetchone()[0]
    assigned = 0
    for condition in conditions.values():
        assigned += _count_rows(condition)

    assert assigned == total


def test_split_conditions_are_disjoint():
    conditions = split_conditions(test_size=0.1, val_size=0.1, seed=42)
    train_keys = _distinct_keys(conditions["train"])
    val_keys = _distinct_keys(conditions["val"])
    test_keys = _distinct_keys(conditions["test"])

    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(test_keys)
    assert all(keys for keys in (train_keys, val_keys, test_keys))


def test_split_conditions_deterministic():
    conditions_a = split_conditions(test_size=0.1, val_size=0.1, seed=42)
    conditions_b = split_conditions(test_size=0.1, val_size=0.1, seed=42)
    conditions_c = split_conditions(test_size=0.1, val_size=0.1, seed=7)

    assert conditions_a == conditions_b
    assert conditions_a != conditions_c


def test_split_conditions_no_val_is_empty():
    conditions = split_conditions(test_size=0.2, val_size=0.0, seed=42)
    assert _count_rows(conditions["val"]) == 0


def test_split_conditions_approximate_fraction():
    con = duckdb.connect()
    total_peptides = con.execute(f"SELECT count(DISTINCT peptidoform) FROM {_source()}").fetchone()[0]
    conditions = split_conditions(test_size=0.1, val_size=0.0, seed=42)
    test_peptides = _distinct_keys(conditions["test"])

    fraction = len(test_peptides) / total_peptides
    assert 0.08 <= fraction <= 0.12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_size": 0.0},
        {"test_size": 1.0},
        {"test_size": 1.1},
        {"val_size": -0.1},
        {"val_size": 1.0},
        {"test_size": 0.6, "val_size": 0.5},
        {"seed": -1},
    ],
)
def test_split_conditions_validation(kwargs):
    with pytest.raises(ValueError):
        split_conditions(**kwargs)


def test_split_peptides_disjoint_and_complete():
    result = split_peptides(TEST_PARQUETS, test_size=0.1, val_size=0.1, seed=42)

    train_set, val_set, test_set = set(result["train"]), set(result["val"]), set(result["test"])
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    con = duckdb.connect()
    all_keys = {row[0] for row in con.execute(
        f"SELECT DISTINCT peptidoform FROM {_source()}"
    ).fetchall()}
    assert train_set | val_set | test_set == all_keys


def test_split_dataset_writes_disjoint_files(tmp_path):
    result = split_dataset(TEST_PARQUETS, tmp_path, test_size=0.1, seed=42)

    assert set(result) == {"train", "test"}
    for paths in result.values():
        assert all(Path(p).exists() for p in paths)

    con = duckdb.connect()
    total = con.execute(f"SELECT count(*) FROM {_source()}").fetchone()[0]
    train_rows = con.execute(
        f"SELECT count(*) FROM parquet_scan('{result['train'][0]}')"
    ).fetchone()[0]
    test_rows = con.execute(
        f"SELECT count(*) FROM parquet_scan('{result['test'][0]}')"
    ).fetchone()[0]
    assert train_rows + test_rows == total
    assert 0.08 <= test_rows / total <= 0.12

    train_keys = {r[0] for r in con.execute(
        f"SELECT DISTINCT peptidoform FROM parquet_scan('{result['train'][0]}')"
    ).fetchall()}
    test_keys = {r[0] for r in con.execute(
        f"SELECT DISTINCT peptidoform FROM parquet_scan('{result['test'][0]}')"
    ).fetchall()}
    assert train_keys.isdisjoint(test_keys)


def test_split_dataset_with_val(tmp_path):
    result = split_dataset(TEST_PARQUETS, tmp_path, test_size=0.1, val_size=0.1, seed=42)
    assert set(result) == {"train", "val", "test"}


def test_split_dataset_refuses_overwrite(tmp_path):
    split_dataset(TEST_PARQUETS, tmp_path, test_size=0.1, seed=42)
    with pytest.raises(FileExistsError):
        split_dataset(TEST_PARQUETS, tmp_path, test_size=0.1, seed=42)

    # overwrite=True must succeed
    result = split_dataset(TEST_PARQUETS, tmp_path, test_size=0.1, seed=42, overwrite=True)
    assert all(Path(p).exists() for p in result["train"])


def test_split_dataset_prefix(tmp_path):
    result = split_dataset(TEST_PARQUETS[0], tmp_path, test_size=0.1, seed=42)
    assert result["train"][0].name.startswith("PXD014877_")

    result2 = split_dataset(TEST_PARQUETS, tmp_path / "multi", test_size=0.1, seed=42)
    assert result2["train"][0].name.startswith("split_")


def test_split_files_partition():
    paths = [Path(f"file_{i}.parquet") for i in range(10)]
    result = split_files(paths, test_size=0.2, val_size=0.3, seed=42)

    assert len(result["test"]) == 2
    assert len(result["val"]) == 3
    assert len(result["train"]) == 5

    all_files = result["train"] + result["val"] + result["test"]
    assert sorted(all_files) == sorted(paths)

    again = split_files(paths, test_size=0.2, val_size=0.3, seed=42)
    assert again == result


def test_loader_extra_where_keeps_split_consistent():
    """The loader must accept a split condition and only serve that split."""
    conditions = split_conditions(test_size=0.1, seed=42)
    test_keys = _distinct_keys(conditions["test"])

    dataset = MS2TorchDataset(TEST_PARQUETS[0], extra_where=conditions["test"])
    loader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=False)

    seen = set()
    for batch in loader:
        seen.update(batch["peptide"])

    assert seen and seen <= test_keys


def test_loader_without_extra_where_unchanged():
    dataset = MS2TorchDataset(TEST_PARQUETS[0], ion_types=("b", "y"))
    loader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    assert isinstance(batch, dict) and len(batch["peptide"]) > 0
