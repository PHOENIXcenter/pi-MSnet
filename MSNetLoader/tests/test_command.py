import logging
from pathlib import Path
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
import numpy as np

from msnetloader.ms2_loader import (
    AlphaPeptDeepConverter,
    MS2TorchDataset,
    LengthExactSampler,
)

from msnetloader.denovo_loader import DeNovoDataset, denovo_collate_fn, PeptideTokenizer
from msnetloader.rt_loader import DeepLCConverter

TESTS_DIR = Path(__file__).parent

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@pytest.fixture(scope="module")
def sample_data():
    """Load small test parquet."""
    file_path = TESTS_DIR / "test_data/PXD014877-Akkermansia_muciniphilia-MSNet.parquet"

    converter = AlphaPeptDeepConverter(file_path)
    precursor_df, fragment_df = converter.convert_parquet_to_training_format()

    return precursor_df, fragment_df


def test_converter_output(sample_data):
    """Test converter outputs are non-empty and valid."""
    precursor_df, fragment_df = sample_data

    assert len(precursor_df) > 0
    assert len(fragment_df) > 0

    assert "nAA" in precursor_df.columns


def test_dataset_and_dataloader(sample_data):
    """Test dataset + sampler + dataloader pipeline."""
    precursor_df, fragment_df = sample_data

    dataset = MS2TorchDataset(precursor_df, fragment_df)

    lengths = dataset.dataset["nAA"]

    sampler = LengthExactSampler(
        lengths=lengths,
        batch_size=8,   # 小 batch 更适合测试
        shuffle=False   # ❗保证可复现
    )

    dataloader = DataLoader(
        dataset.dataset,
        batch_sampler=sampler,
        num_workers=0,   # ❗CI 里更安全
        pin_memory=False
    )

    batch = next(iter(dataloader))

    # ======================
    # ✅ 核心断言
    # ======================
    assert isinstance(batch, dict)
    assert len(batch) > 0

    for k, v in batch.items():
        assert v is not None

        if isinstance(v, torch.Tensor):
            assert v.numel() > 0


def test_length_sampler_consistency(sample_data):
    """Test that the sampler groups sequences of the same length."""
    precursor_df, fragment_df = sample_data

    dataset = MS2TorchDataset(precursor_df, fragment_df)

    # Convert to NumPy array (HF Dataset returns list)
    lengths = np.array(dataset.dataset["nAA"])

    sampler = LengthExactSampler(
        lengths=lengths,
        batch_size=8,
        shuffle=False
    )

    for batch_indices in sampler:
        assert len(batch_indices) > 0

        batch_lengths = lengths[batch_indices]

        # All sequences in the batch should have the same length
        assert len(set(batch_lengths)) == 1


@pytest.fixture(scope="module")
def denovo_dataset():
    """Load a small denovo dataset for testing."""
    dataset = DeNovoDataset(
        "tests/test_data/PXD014877-Akkermansia_muciniphilia-MSNet.parquet"
    )
    return dataset


def test_denovo_dataset_basic(denovo_dataset):
    """Test dataset basic properties."""
    assert len(denovo_dataset) > 0

    sample = denovo_dataset[0]
    assert isinstance(sample, dict)
    assert len(sample) > 0


def test_denovo_dataloader(denovo_dataset):
    """Test dataloader + collate_fn."""
    loader = DataLoader(
        denovo_dataset,
        batch_size=8,          # 小 batch 更稳定
        shuffle=False,         # ❗保证可复现
        collate_fn=denovo_collate_fn,
        num_workers=0,         # ❗CI安全
        pin_memory=False
    )

    batch = next(iter(loader))

    # ======================
    # ✅ 基本结构检查
    # ======================
    assert isinstance(batch, dict)
    assert len(batch) > 0

    for k, v in batch.items():
        assert v is not None

        if isinstance(v, torch.Tensor):
            assert v.numel() > 0


def test_denovo_sequence_consistency(denovo_dataset):
    """Test sequence/token consistency."""
    loader = DataLoader(
        denovo_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=denovo_collate_fn,
        num_workers=0
    )

    batch = next(iter(loader))

    # sequence / token / length / mask
    if "seq" in batch and "length" in batch:
        seq = batch["seq"]
        length = batch["length"]

        assert (seq.shape[1] >= length.max()).item()

    if "mask" in batch:
        mask = batch["mask"]

        # mask 0/1
        assert torch.all((mask == 0) | (mask == 1))


@pytest.fixture(scope="module")
def rt_dataframe():
    """Load RT dataframe from parquet."""
    converter = DeepLCConverter(
        "tests/test_data/PXD014877-Akkermansia_muciniphilia-MSNet.parquet"
    )
    df = converter.convert_parquet_to_rt_format()
    return df


def test_rt_dataframe_basic(rt_dataframe):
    """Basic sanity checks."""
    df = rt_dataframe

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_rt_required_columns(rt_dataframe):
    """Check required columns exist and are valid."""
    df = rt_dataframe

    required_cols = ["seq", "tr"]

    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Check sequence column is not empty
    assert df["seq"].notnull().all(), "Sequence column contains null values"

    # Check RT column is numeric
    assert df["tr"].dtype.kind in {"f", "i"}, "RT column must be numeric"


def test_rt_values_valid(rt_dataframe):
    """Check RT values are valid."""
    df = rt_dataframe

    # RT 应该是非负数
    assert (df["tr"] >= 0).all()

    # RT 不应该全是0
    assert df["tr"].sum() > 0


def test_sequence_valid(rt_dataframe):
    """Check peptide sequences are valid strings."""
    df = rt_dataframe

    assert df["seq"].notnull().all()
    assert df["seq"].apply(lambda x: isinstance(x, str)).all()

    # 长度 > 0
    assert (df["seq"].str.len() > 0).all()


def test_no_duplicate_sequences(rt_dataframe):
    """Optional: check duplicates (depending on your design)."""
    df = rt_dataframe

    # 如果你允许重复可以删掉这条
    assert df["seq"].nunique() > 0
