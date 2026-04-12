import logging
from pathlib import Path
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from msnetloader.ms2_loader import MS2TorchDataset
from msnetloader.denovo_loader import DeNovoIterableDataset
from msnetloader.rt_loader import RTIterableDataset

TESTS_DIR = Path(__file__).parent

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def test_dataset_and_dataloader():
    """Test dataset + sampler + dataloader pipeline."""
    file_path = TESTS_DIR / "test_data/PXD014877-Akkermansia_muciniphilia-MSNet.parquet"
    dataset = MS2TorchDataset(file_path,
                              ion_types=("b, y"))

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
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


def test_denovo_dataloader():
    """Test dataloader + collate_fn."""
    file_path = TESTS_DIR / "test_data/PXD014877-Akkermansia_muciniphilia-MSNet.parquet"

    dataset = DeNovoIterableDataset(file_path)
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
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


def test_rt_dataset_and_dataloader():
    """Test RT dataset + dataloader pipeline."""

    file_path = TESTS_DIR / "test_data/PXD014877-Akkermansia_muciniphilia-MSNet.parquet"

    dataset = RTIterableDataset(
        file_path,
        batch_size=1000,   # 测试时建议小一点
        min_consensus_support=1,
        max_pep=1.0
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False
    )

    batch = next(iter(dataloader))

    # ======================
    # ✅ 基本结构检查
    # ======================
    assert isinstance(batch, dict)
    assert "peptide" in batch
    assert "rt" in batch

    peptides = batch["peptide"]
    rt = batch["rt"]

    # ======================
    # ✅ 非空检查
    # ======================
    assert peptides is not None
    assert rt is not None




