"""Every loader must survive a second pass over the same object.

Each loader used to execute its query once in ``__init__`` and then fetch from
that single-use duckdb cursor inside ``__iter__`` / ``generator``. The first
pass worked, and the second raised ``InvalidInputException: There is no query
result`` — which is what a training loop hits on epoch 2, and what ``tf.data``
hits because it re-invokes the generator on every epoch.
"""

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from msnetloader.denovo_loader import DeNovoIterableDataset
from msnetloader.ms2_loader import MS2TorchDataset
from msnetloader.rt_loader import RTIterableDataset

TESTS_DIR = Path(__file__).parent
CURRENT_FILE = str(TESTS_DIR / "test_data/PXD014877-Sulfolobus_solfataricus-MSNet.parquet")


def _torch_values(loader, key):
    return [value for batch in loader for value in batch[key]]


def _tf_values(dataset, key):
    return [str(value) for batch in dataset.generator() for value in batch[key]]


def _assert_both_passes_match(first, second):
    assert first, "the first pass produced no rows"
    # ORDER BY length(sequence) is not a total order, so equal-length ties may swap
    # between passes; compare the rows as a set rather than by position.
    assert sorted(first) == sorted(second), "the second pass returned different rows"


def test_ms2_torch_dataset_reiterates():
    loader = MS2TorchDataset([CURRENT_FILE], batch_size=32)
    _assert_both_passes_match(_torch_values(loader, "peptide"), _torch_values(loader, "peptide"))


def test_rt_torch_dataset_reiterates():
    loader = RTIterableDataset([CURRENT_FILE], batch_size=32)
    _assert_both_passes_match(_torch_values(loader, "peptide"), _torch_values(loader, "peptide"))


def test_denovo_torch_dataset_reiterates():
    loader = DeNovoIterableDataset([CURRENT_FILE], batch_size=32)
    _assert_both_passes_match(_torch_values(loader, "sequence"), _torch_values(loader, "sequence"))


def test_torch_dataloader_reiterates():
    # The scenario a training loop actually hits: DataLoader re-invokes __iter__
    # once per epoch on the same dataset object.
    loader = MS2TorchDataset([CURRENT_FILE], batch_size=32)
    dataloader = DataLoader(loader, batch_size=None, num_workers=0)

    first = [value for batch in dataloader for value in batch["peptide"]]
    second = [value for batch in dataloader for value in batch["peptide"]]

    _assert_both_passes_match(first, second)


def test_ms2_tf_generator_reiterates():
    pytest.importorskip("tensorflow")
    from msnetloader.ms2_tf import MS2TFDataset

    dataset = MS2TFDataset([CURRENT_FILE], batch_size=32)
    _assert_both_passes_match(_tf_values(dataset, "peptide"), _tf_values(dataset, "peptide"))


def test_rt_tf_generator_reiterates():
    pytest.importorskip("tensorflow")
    from msnetloader.rt_tf import RTTFDataset

    dataset = RTTFDataset([CURRENT_FILE], batch_size=32)
    _assert_both_passes_match(_tf_values(dataset, "peptide"), _tf_values(dataset, "peptide"))


def test_denovo_tf_generator_reiterates():
    pytest.importorskip("tensorflow")
    from msnetloader.denovo_tf import DeNovoTFDataset

    dataset = DeNovoTFDataset([CURRENT_FILE], batch_size=32)

    def sequences():
        # the generator yields (spectrum, sequence, precursor_mz, charge) tuples
        return [str(row[1]) for row in dataset.generator()]

    _assert_both_passes_match(sequences(), sequences())


def test_ms2_tf_dataset_reiterates_across_epochs():
    # tf.data calls the generator callable once per pass, so two epochs over the
    # same tf.data.Dataset must both produce rows.
    pytest.importorskip("tensorflow")
    from msnetloader.ms2_tf import MS2TFDataset

    dataset = MS2TFDataset([CURRENT_FILE], batch_size=32).get_dataset()

    first = [value for batch in dataset.take(1) for value in batch["peptide"]]
    second = [value for batch in dataset.take(1) for value in batch["peptide"]]

    assert len(first) > 0, "the first epoch produced no rows"
    assert len(second) > 0, "the second epoch produced no rows"
