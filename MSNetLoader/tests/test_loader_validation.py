"""Loaders must reject bad arguments instead of silently returning empty targets.

``ion_types`` and ``charges`` used to be filtered against the channel map without
validation, so an unknown value produced a zero-width target tensor (a model
that can never learn anything) and no error at all. ``batch_size`` was passed
straight to duckdb, which answered ``-1`` with an opaque pybind11 ``TypeError``.
"""

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from msnetloader.denovo_loader import DeNovoIterableDataset
from msnetloader.ms2_loader import MS2TorchDataset
from msnetloader.rt_loader import RTIterableDataset

TESTS_DIR = Path(__file__).parent
CURRENT_FILE = str(TESTS_DIR / "test_data/PXD014877-Sulfolobus_solfataricus-MSNet.parquet")


@pytest.mark.parametrize("ion_types", [("z",), ("b", "z"), ("B", "Y")])
def test_ms2_rejects_unknown_ion_type(ion_types):
    with pytest.raises(ValueError, match="Unknown ion_types"):
        MS2TorchDataset([CURRENT_FILE], ion_types=ion_types)


@pytest.mark.parametrize("charges", [(3,), (0,), (-1,)])
def test_ms2_rejects_unknown_charge(charges):
    with pytest.raises(ValueError, match="Unknown charges"):
        MS2TorchDataset([CURRENT_FILE], charges=charges)


def test_ms2_rejects_empty_ion_types():
    with pytest.raises(ValueError, match="ion_types must contain at least one value"):
        MS2TorchDataset([CURRENT_FILE], ion_types=())


def test_ms2_rejects_bare_string_ion_types():
    with pytest.raises(TypeError, match="must be a sequence"):
        MS2TorchDataset([CURRENT_FILE], ion_types="by")


def test_ms2_dedupes_repeated_ion_types():
    dataset = MS2TorchDataset([CURRENT_FILE], ion_types=("b", "b", "y", "y"), charges=(1, 1, 2, 2))

    assert dataset.active_channels == [0, 1, 2, 3]


def test_ms2_preserves_caller_channel_order():
    dataset = MS2TorchDataset([CURRENT_FILE], ion_types=("y", "b"), charges=(1, 2))

    assert dataset.active_channels == [2, 3, 0, 1]


def test_ms2_target_width_matches_active_channels():
    dataset = MS2TorchDataset([CURRENT_FILE], batch_size=16, ion_types=("b",), charges=(1,))
    batch = next(iter(DataLoader(dataset, batch_size=None, num_workers=0)))

    assert dataset.active_channels == [0]
    assert batch["targets"].shape[-1] == 1


def test_ms2_default_target_width_is_four_channels():
    dataset = MS2TorchDataset([CURRENT_FILE], batch_size=16)
    batch = next(iter(DataLoader(dataset, batch_size=None, num_workers=0)))

    assert dataset.active_channels == [0, 1, 2, 3]
    assert batch["targets"].shape[-1] == 4


@pytest.mark.parametrize("batch_size", [0, -1, 2.5])
@pytest.mark.parametrize(
    "make_loader",
    [
        lambda batch_size: MS2TorchDataset([CURRENT_FILE], batch_size=batch_size),
        lambda batch_size: RTIterableDataset([CURRENT_FILE], batch_size=batch_size),
        lambda batch_size: DeNovoIterableDataset([CURRENT_FILE], batch_size=batch_size),
    ],
    ids=["ms2", "rt", "denovo"],
)
def test_torch_loaders_reject_invalid_batch_size(make_loader, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        make_loader(batch_size)


@pytest.mark.parametrize("batch_size", [0, -1, 2.5])
@pytest.mark.parametrize(
    "module_and_class",
    [
        ("ms2_tf", "MS2TFDataset"),
        ("rt_tf", "RTTFDataset"),
        ("denovo_tf", "DeNovoTFDataset"),
    ],
    ids=["ms2_tf", "rt_tf", "denovo_tf"],
)
def test_tf_loaders_reject_invalid_batch_size(module_and_class, batch_size):
    pytest.importorskip("tensorflow")
    module = pytest.importorskip(f"msnetloader.{module_and_class[0]}")
    dataset_class = getattr(module, module_and_class[1])

    with pytest.raises(ValueError, match="batch_size"):
        dataset_class([CURRENT_FILE], batch_size=batch_size)


def test_ms2_tf_rejects_unknown_ion_type():
    pytest.importorskip("tensorflow")
    from msnetloader.ms2_tf import MS2TFDataset

    with pytest.raises(ValueError, match="Unknown ion_types"):
        MS2TFDataset([CURRENT_FILE], ion_types=("z",))


def test_ms2_tf_dedupes_repeated_ion_types():
    pytest.importorskip("tensorflow")
    from msnetloader.ms2_tf import MS2TFDataset

    dataset = MS2TFDataset([CURRENT_FILE], ion_types=("b", "b", "y", "y"), charges=(1, 1, 2, 2))

    assert dataset.active_channels == [0, 1, 2, 3]
