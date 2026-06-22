import math
import os.path
# from koinapy import Koina
import numpy as np
import torch.nn.parallel
from peptdeep.settings import global_settings
import itertools
import pandas as pd
import re
import functools
import alphabase.peptide.fragment as fragment
from glob import glob
import json
from peptdeep.model.ms2 import ModelMS2pDeep, pDeepModel, calc_ms2_similarity, ModelMS2Transformer, ModelMS2Bert
from pandarallel import pandarallel  # import pandarallel
import gc
from sklearn.metrics.pairwise import cosine_similarity
from peptdeep.utils import get_available_device
from lightning.pytorch.strategies import DDPStrategy
from alphabase.peptide.fragment import get_sliced_fragment_dataframe
from torch.utils.data import Dataset
import lightning.pytorch as pl
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import ConcatDataset, BatchSampler
import random
import torch.distributed as dist
import logging
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.utilities import rank_zero_only
from peptdeep.model.featurize import parse_instrument_indices

color = ["#FFD2EB", "#D2FFD3", "#C7EAFF", "#FFE0A8"]

logger = logging.getLogger("MS2")
torch.set_printoptions(threshold=np.inf)


def modification_conversion(row):
    mod_names = []
    if pd.isna(row["modifications"]):
        mod_sites = ""
    else:
        mod_sites = ";".join([m.split("-")[0] for m in eval(row["modifications"])])
        for m in eval(row["modifications"]):
            sites = "Protein_N-term" if int(m.split("-")[0]) == 0 else row["sequence"][int(m.split("-")[0]) - 1]
            mod_names.append(m.split("-")[1].split(" ")[0] + "@" + sites)

    nAA = len(row["sequence"])
    return ";".join(mod_names), mod_sites, nAA


def ion_matched(row):
    pred_int = len(row["sequence"])
    b_z1 = []
    b_z2 = []
    y_z1 = []
    y_z2 = []
    ions = row["ions_matched"]
    all_matched_intensity = sum([float(ion.split(",")[1]) for ion in ions])
    for i in range(1, pred_int):
        b_z1_pattern = "b" + str(i) + "/"
        matches = list(filter(lambda ion: re.search(b_z1_pattern, ion), ions))
        if len(matches) > 0:
            result = max([float(match.split(",")[1]) for match in matches])
        else:
            result = 0
        b_z1.append(result)

        b_z2_pattern = "b" + str(i) + "\^2/"
        matches = list(filter(lambda ion: re.search(b_z2_pattern, ion), ions))
        if len(matches) > 0:
            result = max([float(match.split(",")[1]) for match in matches])
        else:
            result = 0
        b_z2.append(result)

        y_z1_pattern = "y" + str(i) + "/"
        matches = list(filter(lambda ion: re.search(y_z1_pattern, ion), ions))
        if len(matches) > 0:
            result = max([float(match.split(",")[1]) for match in matches])
        else:
            result = 0
        y_z1.append(result)

        y_z2_pattern = "y" + str(i) + "\^2/"
        matches = list(filter(lambda ion: re.search(y_z2_pattern, ion), ions))
        if len(matches) > 0:
            result = max([float(match.split(",")[1]) for match in matches])
        else:
            result = 0
        y_z2.append(result)

    max_int = max(b_z1 + b_z2 + y_z1 + y_z2)
    if max_int == 0:
        return 0, 0, 0, 0, 'low'

    # count_non_zero = len(list(filter(lambda x: x != 0, b_z1 + b_z2 + y_z1 + y_z2)))

    if all_matched_intensity < 0.10 * sum(eval(row["intensity_array"])):
        return list(map(lambda x: x / max_int, b_z1)), list(map(lambda x: x / max_int, b_z2)), list(
            map(lambda x: x / max_int, y_z1)), list(map(lambda x: x / max_int, y_z2)), 'low'
    else:
        return list(map(lambda x: x / max_int, b_z1)), list(map(lambda x: x / max_int, b_z2)), list(
            map(lambda x: x / max_int, y_z1)), list(map(lambda x: x / max_int, y_z2)), 'high'


class MSNetMS2DataSets(Dataset):
    def __init__(self, precursor_df: pd.DataFrame, fragment_df: pd.DataFrame, random_state: Optional[int] = None):
        super().__init__()
        self.precursor_df = precursor_df
        self.fragment_df = fragment_df
        self.rng = np.random.default_rng(random_state)

    def __getitem__(self, idx: int):
        """
        Return the annotated MS/MS spectrum with the given index.

        Parameters
        ----------
        idx : int
            The index of the spectrum to return.

        Returns
        -------
        spectrum : torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.
        annotation : str
            The peptide annotation of the spectrum.
        """
        batch_df = self.precursor_df.iloc[idx]

        return batch_df, self.fragment_df

    def __len__(self):
        return len(self.precursor_df)

    @property
    def rng(self):
        """The NumPy random number generator."""
        return self._rng

    @rng.setter
    def rng(self, seed):
        """Set the NumPy random number generator."""
        self._rng = np.random.default_rng(seed)


class GroupedBatchSampler(BatchSampler):
    def __init__(self, dataset_to_indices, batch_size, shuffle=True):
        self.batches = []
        self.shuffle = shuffle

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        all_batches = []
        for indices in dataset_to_indices:
            if self.shuffle:
                random.shuffle(indices)

            # Partition samples of the current dataset into batches
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                if len(batch) == batch_size:
                    all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        # Only take batches assigned to the current rank
        self.batches = all_batches[self.rank::self.world_size]

    def __iter__(self):
        # for b in self.batches:
        #     print("Batch indices:", b)
        #     assert all(isinstance(i, int) for i in b), f"Invalid batch: {b}"

        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            precursor_df: pd.DataFrame = None,
            fragment_df: pd.DataFrame = None,
            train_batch_size: int = 1024,
            n_workers: Optional[int] = None,
            random_state: Optional[int] = None,
    ):
        super().__init__()
        self.precursor_df = precursor_df
        self.fragment_df = fragment_df
        self.train_batch_size = train_batch_size
        # self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.n_workers = 2
        self.rng = np.random.default_rng(random_state)
        self.train_dataset = None
        self.concat_dataset = None

    def setup(self, stage=None) -> None:
        """
        Set up the PyTorch Datasets.

        Parameters
        ----------
        stage : str {"fit", "validate", "test"}
            The stage indicating which Datasets to prepare. All are prepared by
            default.
        annotated: bool
            True if peptide sequence annotations are available for the test
            data.
        """
        _grouped = list(self.precursor_df.sample(frac=1).groupby("nAA"))
        rnd_nAA = np.random.permutation(len(_grouped))
        make_dataset = functools.partial(  # Create new function: AnnotatedSpectrumDataset
            MSNetMS2DataSets
        )
        # Merge all precursor_df and record batch indices (nAA is consistent within each batch)
        all_precursor_rows = []
        self.dataset_to_indices = []
        current_index = 0

        for group_df in rnd_nAA:
            nAA, df_group = _grouped[group_df]
            group_df = df_group.sample(frac=1, random_state=42)  # Shuffle within each group again

            num_rows = len(group_df)
            num_full_batches = num_rows // self.train_batch_size

            for i in range(num_full_batches):
                batch_df = group_df.iloc[i * self.train_batch_size: (i + 1) * self.train_batch_size]
                all_precursor_rows.append(batch_df)
                batch_indices = list(range(current_index, current_index + len(batch_df)))
                self.dataset_to_indices.append(batch_indices)
                current_index += len(batch_df)

        # Merge all samples into one large dataframe, build dataset
        combined_precursor_df = pd.concat(all_precursor_rows, ignore_index=True)
        self.train_dataset = make_dataset(combined_precursor_df, self.fragment_df)

    def _make_loader(
            self,
            dataset: torch.utils.data.Dataset,
            batch_size: int,
            shuffle: bool = False,
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            A PyTorch Dataset.
        batch_size : int
            The batch size to use.
        shuffle : bool
            Option to shuffle the batches.

        Returns
        -------
        torch.utils.data.DataLoader
            A PyTorch DataLoader.
        """
        sampler = GroupedBatchSampler(self.dataset_to_indices, batch_size=batch_size, shuffle=True)
        #
        # print("Dataset len:", len(dataset))
        # print("Sample idx 0:", dataset[0])  # inspect return value structure

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=self.prepare_batch,
            sampler=sampler,
            pin_memory=True,
            num_workers=self.n_workers,
            # shuffle=shuffle,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""
        return self._make_loader(
            self.train_dataset, self.train_batch_size, shuffle=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the validation DataLoader."""
        return self._make_loader(self.valid_dataset, self.eval_batch_size)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the test DataLoader."""
        return self._make_loader(self.test_dataset, self.eval_batch_size)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the predict DataLoader."""
        return self._make_loader(self.test_dataset, self.eval_batch_size)

    def prepare_batch(self,
                      batch
                      ):
        """
        Collate MS/MS spectra into a batch.

        The MS/MS spectra will be padded so that they fit nicely as a tensor.
        However, the padded elements are ignored during the subsequent steps.

        Parameters
        ----------
        batch : List[Tuple[torch.Tensor, float, int, str]]
            A batch of data from an AnnotatedSpectrumDataset, consisting of for each
            spectrum (i) a tensor with the m/z and intensity peak values, (ii), the
            precursor m/z, (iii) the precursor charge, (iv) the spectrum identifier.

        Returns
        -------
        spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
            The padded mass spectra tensor with the m/z and intensity peak values
            for each spectrum.
        precursors : torch.Tensor of shape (batch_size, 3)
            A tensor with the precursor neutral mass, precursor charge, and
            precursor m/z.
        spectrum_ids : np.ndarray
            The spectrum identifiers (during de novo sequencing) or peptide
            sequences (during training).
        """

        batch_df, fragment_intensity_df = batch
        targets = torch.tensor(
            get_sliced_fragment_dataframe(
                fragment_intensity_df,
                batch_df[["frag_start_idx", "frag_stop_idx"]].values,
            ).values
            , dtype=torch.float32).view(-1, batch_df.nAA.values[0] - 1, 4)

        # features = torch.tensor(
        #     get_ascii_indices(batch_df["sequence"].values.astype("U")), dtype=torch.long
        # )

        return targets, batch_df


class MSNetMS2Model(pl.LightningModule, pDeepModel):
    def __init__(self):
        pl.LightningModule.__init__(self)
        pDeepModel.__init__(self)
        self._history = []
        self.n_log = 1211
        self.tb_summarywriter = SummaryWriter("./DDP")

    def _get_26aa_indice_features(self, batch_df: pd.DataFrame) -> torch.LongTensor:
        """
        Get indices values for 26 upper-case letters (amino acids),
        from 1 to 26. 0 is used for padding.
        """
        return self._as_tensor(
            self.get_batch_aa_indices(batch_df["sequence"].values.astype("U")),
            dtype=torch.long,
        )

    def get_batch_aa_indices(self, seq_array: Union[List, np.ndarray]) -> np.ndarray:
        """
        Convert peptide sequences into AA ID array. ID=0 is reserved for masking,
        so ID of 'A' is 1, ID of 'B' is 2, ..., ID of 'Z' is 26 (maximum).
        Zeros are padded into the N- and C-term for each sequence.

        Parameters
        ----------
        seq_array : Union[List,np.ndarray]
            list or 1-D array of sequences with the same length

        Returns
        -------
        np.ndarray
            2-D `np.int32` array with the shape
            `(len(seq_array), len(seq_array[0])+2)`. Zeros is padded into the
            N- and C-term of each sequence, so the 1st-D is `len(seq_array[0])+2`.

        """
        x = np.array(seq_array).view(np.int32).reshape(len(seq_array), -1) - ord("A") + 1
        # padding zeros at the N- and C-term
        return np.pad(x, [(0, 0)] * (len(x.shape) - 1) + [(1, 1)])

    def _get_features_from_batch_df(
            self,
            batch_df: pd.DataFrame,
            **kwargs,
    ) -> Tuple[torch.Tensor]:
        aa_indices = self._get_26aa_indice_features(batch_df)

        mod_x = self._get_mod_features(batch_df)

        charges = (
                self._as_tensor(batch_df["charge"].values).unsqueeze(1) * self.charge_factor
        )

        nces = self._as_tensor(batch_df["nce"].values).unsqueeze(1) * self.NCE_factor

        instrument_indices = self._as_tensor(
            parse_instrument_indices(batch_df["instrument"]), dtype=torch.long
        )
        return aa_indices, mod_x, charges, nces, instrument_indices

    def training_step(self, batch):
        targets, batch_df = batch
        features = self._get_features_from_batch_df(batch_df=batch_df)
        predicts = self.model(*features)

        # get_batch_aa_indices(batch_df["sequence"].values.astype("U"))
        cost = self.loss_func(predicts, targets)
        self.log(
            f"train_Loss",
            cost.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return cost

    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        train_loss = self.trainer.callback_metrics["train_Loss"].detach()
        self.log("train_Loss", self.trainer.callback_metrics["train_Loss"])
        metrics = {
            "step": self.trainer.global_step,
            "train": train_loss.item(),
        }

        self._history.append(metrics)

        self._log_history()

    @rank_zero_only
    def _log_history(self) -> None:
        """
        Write log to console, if requested.
        """
        # Log only if all output for the current epoch is recorded.
        if len(self._history) == 0:
            return
        if len(self._history) == 1:
            header = "Step\tloss\t"
            logger.info(header)
        metrics = self._history[-1]
        if metrics["step"] % self.n_log == 0:
            msg = "%i\t%.6f"
            vals = [
                metrics["step"],
                metrics.get("train", np.nan),
            ]

            logger.info(msg, *vals)
            if self.tb_summarywriter is not None:
                for descr, key in [
                    ("loss/train_loss", "train"),
                ]:
                    metric_value = metrics.get(key, np.nan)
                    if not np.isnan(metric_value):
                        self.tb_summarywriter.add_scalar(
                            descr, metric_value, metrics["step"]
                        )

    def configure_optimizers(
            self,
    ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=22, verbose=True,
        #                                            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
        #                                            eps=1e-08)

        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, 20 * 1211, 100 * 1211 - 20 * 1211
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}

    def on_train_end(self):
        # Manually save checkpoint
        torch.save(self.model.state_dict(), "DDP/scaling_law_model_P4.pt")


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm-up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the learning rate.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_iters: int,
            cosine_schedule_period_iters: int,
    ):
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        # print([base_lr * lr_factor for base_lr in self.base_lrs])
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):  # epoch returns the number of batches processed
        # print(epoch)
        lr_factor = 0.5 * (
                1 + np.cos(np.pi * epoch / self.cosine_schedule_period_iters)
        )
        if epoch <= self.warmup_iters:
            lr_factor *= epoch / self.warmup_iters
        # print(lr_factor)
        return lr_factor


def build_train_data(dir_match, pre_len):
    print(dir_match)
    if "Cohort_E480_DDAQC" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/Cohort_E480_DDAQC_precursor_df_top20.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('Cohort_E480_DDAQC_precursor_df_top20.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/Cohort_E480_DDAQC_fragment_intensity_df_top20.csv")
    elif "PXD012131" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD012131_precursor_df_top20.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD012131_precursor_df_top20.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD012131_fragment_intensity_df_top20.csv")
    elif "PXD014877_msnet_Drosophila" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD014877_msnet_Drosophila_precursor_df_top20.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD014877_precursor_df_top20.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD014877_msnet_Drosophila_fragment_intensity_df_top20.csv")
    elif "PXD006675" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD006675_precursor_df_top20.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD006675_precursor_df_top20.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD006675_fragment_intensity_df_top20.csv")
    elif "PXD010899" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD010899_precursor_df_top20.csv')
        # IPX0001804001_df["nce"] = 30
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD010899_precursor_df_top20.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD010899_fragment_intensity_df_top20.csv")
    elif "PXD030983_msnet" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD030983_msnet_precursor_df_top20.csv')
        IPX0001804001_df["nAA"] = IPX0001804001_df.apply(lambda row: len(row["sequence"]), axis=1)
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD030983_precursor_df_top20.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD030983_msnet_fragment_intensity_df_top20.csv")
    elif "PXD004732" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD004732_precursor_df.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD004732_precursor_df.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        # fragment_intensity_df.to_csv("PXD030983_fragment_intensity_df.csv", index=False)
        fragment_intensity_df = pd.read_csv("train_data/PXD004732_fragment_intensity_df.csv")
    elif "PXD010595" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD010595_precursor_df.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD010595_precursor_df.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        # fragment_intensity_df.to_csv("PXD030983_fragment_intensity_df.csv", index=False)
        fragment_intensity_df = pd.read_csv("train_data/PXD010595_fragment_intensity_df.csv")
    elif "PXD004242" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD004242_precursor_df_top20.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD004242_precursor_df_top20.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        # fragment_intensity_df.to_csv("PXD030983_fragment_intensity_df.csv", index=False)
        fragment_intensity_df = pd.read_csv("train_data/PXD004242_fragment_intensity_df_top20.csv")
    elif "PXD008722" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD008722_precursor_df_top20.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD008722_precursor_df_top20_test.csv', index=False)
        # IPX0001804001_df.fillna("", inplace=True)
        # fragment_intensity_df.to_csv("PXD030983_fragment_intensity_df.csv", index=False)
        fragment_intensity_df = pd.read_csv("train_data/PXD008722_fragment_intensity_df_top20.csv")
    elif "PXD008840" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD008840_precursor_df_top20.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD008840_precursor_df_top20.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        # fragment_intensity_df.to_csv("PXD030983_fragment_intensity_df.csv", index=False)
        fragment_intensity_df = pd.read_csv("train_data/PXD008840_fragment_intensity_df_top20.csv")
    elif "PXD028735" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD028735-Ecoli_precursor_df.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD028735-Ecoli_precursor_df.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        # fragment_intensity_df.to_csv("PXD030983_fragment_intensity_df.csv", index=False)
        fragment_intensity_df = pd.read_csv("train_data/PXD028735-Ecoli_fragment_intensity_df.csv")
    elif "PXD004452" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD004452_precursor_df.csv')
        # IPX0001804001_df['nce'] = 28
        # IPX0001804001_df['instrument'] = 'QEHF'
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD004452_precursor_df.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        # fragment_intensity_df.to_csv("PXD030983_fragment_intensity_df.csv", index=False)
        fragment_intensity_df = pd.read_csv("train_data/PXD004452_fragment_intensity_df.csv")
    elif "PXD024364" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD024364_precursor_df.csv')
        IPX0001804001_df["nce"] = IPX0001804001_df["nce"].astype(int)
        # IPX0001804001_df['instrument'] = 'Lumos'
        t = IPX0001804001_df["nAA"].apply(lambda x: int(x) - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD024364_precursor_df.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        # fragment_intensity_df.to_csv("PXD030983_fragment_intensity_df.csv", index=False)
        fragment_intensity_df = pd.read_csv("train_data/PXD024364_fragment_intensity_df.csv")
    elif "PXD010154" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD010154_precursor_df_top20.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        IPX0001804001_df["nce"] = IPX0001804001_df["nce"].astype(int)
        # IPX0001804001_df.to_csv('PXD010154_precursor_df_top20.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD010154_fragment_intensity_df_top20.csv")
    elif "PXD021013" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD021013_msnet_precursor_df.csv')
        IPX0001804001_df.replace("Orbitrap Fusion Lumos", 'Lumos', inplace=True)
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        IPX0001804001_df["nce"] = IPX0001804001_df["nce"].astype(int)
        # IPX0001804001_df.to_csv('PXD021013_msnet_precursor_df.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD021013_msnet_fragment_intensity_df.csv")
    elif "PXD013868" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD013868_precursor_df_top20.csv')
        # IPX0001804001_df.replace("Q Exactive HF", 'QEHF', inplace=True)
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD013868_msnetprot_precursor_df_top20.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD013868_fragment_intensity_df_top20.csv")
    elif "PXD019643" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD019643_precursor_df.csv')
        IPX0001804001_df["nAA"] = IPX0001804001_df.apply(lambda row: len(row["sequence"]), axis=1)
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD019643_precursor_df.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD019643_fragment_intensity_df.csv")
    elif "PXD012636_horse_msnet" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD012636_horse_msnet_precursor_df.csv')
        # IPX0001804001_df.replace("Q Exactive HF", 'QEHF', inplace=True)
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD012636_horse_msnet_precursor_df.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD012636_horse_msnet_fragment_intensity_df.csv")
    elif "PXD012636_mouse_msnet" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD012636_mouse_msnet_precursor_df.csv')
        # IPX0001804001_df.replace("Q Exactive HF", 'QEHF', inplace=True)
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD012636_mouse_msnet_precursor_df.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD012636_mouse_msnet_fragment_intensity_df.csv")
    elif "PXD012636_Pig_msnet" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD012636_Pig_msnet_precursor_df.csv')
        # IPX0001804001_df.replace("Q Exactive HF", 'QEHF', inplace=True)
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        # IPX0001804001_df.to_csv('PXD012636_Pig_msnet_precursor_df.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD012636_Pig_msnet_fragment_intensity_df.csv")
    elif "PXD012636_Rat_msnet" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD012636_Rat_msnet_precursor_df.csv')
        # IPX0001804001_df.replace("Q Exactive HF", 'QEHF', inplace=True)
        # IPX0001804001_df.replace("Q Exactive Plus", 'QE', inplace=True)
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]

        # IPX0001804001_df.to_csv('PXD012636_Rat_msnet_precursor_df.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD012636_Rat_msnet_fragment_intensity_df.csv")
    elif "IPX0001804001" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/IPX0001804001_precursor_df_top20.csv')
        # IPX0001804001_df.replace("Fusion", 'Lumos', inplace=True)
        # IPX0001804001_df.replace("Q Exactive Plus", 'QE', inplace=True)
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        IPX0001804001_df["nce"] = IPX0001804001_df["nce"].astype(int)
        # IPX0001804001_df.to_csv('IPX0001804001_precursor_df.csv', index=False)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/IPX0001804001_fragment_intensity_df_top20.csv")
    elif "PXD002767" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD002767_precursor_df.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        IPX0001804001_df["nce"] = IPX0001804001_df["nce"].astype(int)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD002767_fragment_intensity_df.csv")
    elif "PXD014877_msnet_Triticum_aestivum" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD014877_msnet_Triticum_aestivum_precursor_df.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        IPX0001804001_df["nce"] = IPX0001804001_df["nce"].astype(int)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD014877_msnet_Triticum_aestivum_fragment_intensity_df.csv")
    elif "IPX0002031000" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/IPX0002031000_precursor_df_top20.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        IPX0001804001_df["nce"] = IPX0001804001_df["nce"].astype(int)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/IPX0002031000_fragment_intensity_df_top20.csv")
    elif "PXD014877_Gossypium_msnet" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD014877_Gossypium_msnet_precursor_df.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        IPX0001804001_df["nce"] = IPX0001804001_df["nce"].astype(int)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD014877_Gossypium_msnet_fragment_intensity_df.csv")
    elif "PXD000865" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD000865_msnet_notrysin_precursor_df.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        IPX0001804001_df["nce"] = IPX0001804001_df["nce"].astype(int)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD000865_msnet_notrysin_fragment_intensity_df.csv")
    elif "PXD002395" in dir_match:
        IPX0001804001_df = pd.read_csv('train_data/PXD002395_precursor.csv')
        t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
        end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
        IPX0001804001_df['frag_stop_idx'] = end
        IPX0001804001_df['frag_start_idx'] = [pre_len] + end[:-1]
        IPX0001804001_df["nce"] = IPX0001804001_df["nce"].astype(int)
        IPX0001804001_df.fillna("", inplace=True)
        fragment_intensity_df = pd.read_csv("train_data/PXD002395_fragment_intensity.csv")
    # else:
    #     base_name = dir_match.split("/")[1]
    #     IPX0001804001_data = []
    #     identifications = glob(dir_match)
    #     print(len(identifications))
    #     for j in identifications:
    #         data = pd.read_parquet(j, columns=["sequence", "charge", "is_decoy", "ions_matched", "modifications",
    #                                            "intensity_array", "USI"])
    #         data = data[data["is_decoy"] == 0]
    #         try:
    #             meta = pd.read_parquet(j.replace(".parquet", "_meta.parquet"),
    #                                    columns=["Instrument", "USI", "Collision Energy"])
    #         except Exception as e:
    #             print(e)
    #             continue
    #         data = pd.merge(data, meta, on="USI", how="inner")
    #         if data["sequence"].isnull().any():
    #             print(j)
    #             data.dropna(subset=["sequence"], inplace=True)
    #         IPX0001804001_data.append(data)
    #
    #     IPX0001804001_df = pd.concat(IPX0001804001_data, axis=0, ignore_index=True)
    #     del IPX0001804001_data
    #     gc.collect()
    #     IPX0001804001_df[["mods", "mod_sites", 'nAA']] = IPX0001804001_df.apply(
    #         lambda row: modification_conversion(row),
    #         result_type='expand', axis=1)
    #     IPX0001804001_df = IPX0001804001_df[-IPX0001804001_df["mods"].str.contains("Gln")]
    #     IPX0001804001_df = IPX0001804001_df[IPX0001804001_df["nAA"] > 6]
    #     print("OK")
    #     splits = np.array_split(IPX0001804001_df, 10)
    #     IPX0001804001_df = []
    #     # splits is a list containing 10 DataFrames
    #     for i, part in enumerate(splits):
    #         pandarallel.initialize(nb_workers=10)
    #         print(f"Part {i}: {len(part)} rows")
    #         part[['b_z1', 'b_z2',
    #               'y_z1', 'y_z2', 'quality']] = part.parallel_apply(lambda row: ion_matched(row),
    #                                                                 axis=1,
    #                                                                 result_type='expand')
    #         IPX0001804001_df.append(part)
    #
    #     IPX0001804001_df = pd.concat(IPX0001804001_df, ignore_index=True)
    #     IPX0001804001_df = IPX0001804001_df[(IPX0001804001_df["b_z1"] != 0) & (IPX0001804001_df["quality"] == 'high')]
    #     # IPX0001804001_df["Collision Energy"] = 27
    #     # IPX0001804001_df["Instrument"] = "Q Exactive HF-X"
    #     IPX0001804001_df.rename(columns={'Instrument': "instrument", "Collision Energy": "nce"}, inplace=True)
    #     t = IPX0001804001_df["nAA"].apply(lambda x: x - 1)
    #     end = [i + pre_len for i in list(itertools.accumulate(t.values.tolist()))]
    #     IPX0001804001_df['frag_stop_idx'] = end
    #     IPX0001804001_df['frag_start_idx'] = [0] + end[:-1]
    #     IPX0001804001_df.rename(columns={'Instrument': "instrument", "Collision Energy": "nce"}, inplace=True)
    #     fragment_intensity_df = pd.DataFrame()
    #     fragment_intensity_df['b_z1'] = IPX0001804001_df[['b_z1']].explode('b_z1')
    #     fragment_intensity_df['b_z2'] = IPX0001804001_df[['b_z2']].explode('b_z2')
    #     fragment_intensity_df['y_z1'] = IPX0001804001_df[['y_z1']].explode('y_z1')
    #     fragment_intensity_df['y_z2'] = IPX0001804001_df[['y_z2']].explode('y_z2')
    #     fragment_intensity_df = fragment_intensity_df.reset_index(drop=True)
    #
    #     IPX0001804001_df.drop(["intensity_array", "ions_matched", "modifications",
    #                            "b_z1", "b_z2", "y_z1", "y_z2", "quality", "USI"], axis=1, inplace=True)
    #
    #     IPX0001804001_df.to_csv('./train_data/{0}_precursor_df.csv'.format(base_name), index=False)
    #     fragment_intensity_df.to_csv("./train_data/{0}_fragment_intensity_df.csv".format(base_name), index=False)

    # print(IPX0001804001_df[IPX0001804001_df["nce"] == 2.0])
    # print(IPX0001804001_df[IPX0001804001_df["nce"] == ""])
    print(IPX0001804001_df.shape[0])  # 5689996
    print(IPX0001804001_df.drop_duplicates(subset=["sequence", "charge", "mods", "mod_sites"]).shape[0])
    return IPX0001804001_df, fragment_intensity_df


def train():
    # only compare missing peak correlation
    train_data = [
        "../PXD004732/PXD0004732_msnet/ionmatched/*_psm.parquet",  # 7934487 OK
        "../PXD010595/PXD010595_msnet/ionmatched/*_psm_clean.parquet",  # 5459100 OK
        "../Cohort_E480_DDAQC/huiyan_msnet/ionmatched/*_psm.parquet",  # 5631504 OK
        "../PXD012131/ionmatched/*_psm.parquet",  # 2961199 OK
        "../PXD014877/PXD014877_msnet_Drosophila/ionmatched/*_psm.parquet",  # 114172
        "../PXD006675/PXD006675_msnet/ionmatched/*_psm.parquet",  # 8340318 OK
        "../PXD010899/PXD010899_lfq_msnet/ionmatched/*_psm.parquet",  # 1249685 OK
        "../PXD030983_msnet/ionmatched/*_psm.parquet",  # 7988744
        "../PXD004242/PXD004242_msnet/ionmatched/*_psm.parquet",  # 4785586
        "../PXD008722/PXD008722_msnet/ionmatched/*_psm.parquet",  # 2887038
        "../PXD008840/PXD008840_msnet/ionmatched/*_psm.parquet",  # 901947     # 1316200
        "../PXD028735-Ecoli/ionmatched/*_psm.parquet",  # 61768
        "../PXD004452/PXD004452-Gluc_msnet/ionmatched/*_psm.parquet",  # 119186
        "../PXD024364/PXD024364_lysc_msnet/ionmatched/*_psm.parquet",  # 384150
        "../PXD021013_msnet/ionmatched/*_psm.parquet",  # OK
        "../PXD013868_msnetprot/ionmatched/*_psm.parquet",  # OK
        "../PXD019643_msgf_comet_only/ionmatched/*_psm.parquet",  # failed to run
        "../PXD012636_horse_msnet/ionmatched/*_psm.parquet",  # OK
        "../PXD012636_mouse_msnet/ionmatched/*_psm.parquet",  # OK
        "../PXD012636_Pig_msnet/ionmatched/*_psm.parquet",  # OK
        "../PXD012636_Rat_msnet/ionmatched/*_psm.parquet",  # OK
        "../PXD010154/PXD010154-part1/ionmatched/*_psm.parquet",
        "../IPX0001804001/ionmatched/*_psm.parquet",
        "../PXD002767/PXD002767_msnet/ionmatched/*_psm.parquet",
        "../PXD014877/PXD014877_msnet_Triticum_aestivum/ionmatched/*_psm.parquet",
        "../IPX0002031000/ionmatched/*_psm.parquet",
        "../PXD014877/PXD014877_Gossypium_msnet/ionmatched/*_psm.parquet",
        "../PXD000865_msnet_notrysin/ionmatched/*_psm.parquet",
        "../PXD002395"
    ]

    train_ms2_precursor = []
    train_intensity_df = []
    pre_len = 0
    for td in train_data:
        precursor_df, fragment_intensity_df = build_train_data(td, pre_len)
        pre_len += fragment_intensity_df.shape[0]
        train_ms2_precursor.append(precursor_df)
        train_intensity_df.append(fragment_intensity_df)
    precursor_df = pd.concat(train_ms2_precursor, ignore_index=True)
    fragment_intensity_df = pd.concat(train_intensity_df, ignore_index=True)
    fragment_intensity_df.reset_index(drop=True, inplace=True)
    precursor_df.drop(["is_decoy"], axis=1, inplace=True)
    precursor_df.replace("Orbitrap Fusion Lumos", "Lumos", inplace=True)
    precursor_df.replace("Q Exactive HF", "QE", inplace=True)
    precursor_df.replace("Q Exactive HF-X", "QE", inplace=True)
    precursor_df.replace("Q Exactive Plus", "QE", inplace=True)
    precursor_df.replace("Q Exactive", "QE", inplace=True)
    precursor_df.replace("Fusion", "Lumos", inplace=True)
    precursor_df.replace("Orbitrap Fusion", "Lumos", inplace=True)
    precursor_df.replace("EXPLORIS480", "QE", inplace=True)
    precursor_df.replace("QEHFX", "QE", inplace=True)
    precursor_df.replace("QEHF", "QE", inplace=True)
    precursor_df.replace("LTQ Orbitrap Velos", "Lumos", inplace=True)
    precursor_df.replace("LTQ Orbitrap Elite", "Lumos", inplace=True)
    precursor_df["mods"] = precursor_df.apply(lambda row: row["mods"].replace("Glu@E", "Glu->pyro-Glu@E^Any_N-term"),
                                              axis=1)
    precursor_df["nce"] = precursor_df["nce"].astype(int)
    precursor_df["frag_start_idx"] = precursor_df["frag_start_idx"].astype(int)
    precursor_df["frag_stop_idx"] = precursor_df["frag_stop_idx"].astype(int)
    modss = precursor_df.apply(lambda row: row["mods"].split(";"), axis=1)
    print(set([aa for sublist in modss for aa in sublist]))
    print(set("".join(precursor_df["sequence"].unique())))
    print(precursor_df["instrument"].unique())
    print(precursor_df["nce"].unique())
    print(precursor_df.shape)  # 49232392
    unique_precursors = precursor_df[["sequence", "charge", "mods", "mod_sites"]].drop_duplicates()
    print(unique_precursors.shape[0])  # 1233485
    unique_precursors = unique_precursors.sample(frac=0.5, random_state=42).reset_index(drop=True)
    result = pd.merge(unique_precursors, precursor_df, how="left", on=["sequence", "charge", "mods", "mod_sites"])
    result.to_csv("train_data/scaling_law_P1.csv", index=False)
    print(result.shape)
    
    unique_precursors = unique_precursors.sample(frac=0.5, random_state=42).reset_index(drop=True)
    result = pd.merge(unique_precursors, precursor_df, how="left", on=["sequence", "charge", "mods", "mod_sites"])
    result.to_csv("train_data/scaling_law_P2.csv", index=False)
    print(result.shape)
    
    unique_precursors = unique_precursors.sample(frac=0.5, random_state=42).reset_index(drop=True)
    result = pd.merge(unique_precursors, precursor_df, how="left", on=["sequence", "charge", "mods", "mod_sites"])
    result.to_csv("train_data/scaling_law_P3.csv", index=False)
    print(result.shape)
    
    unique_precursors = unique_precursors.sample(frac=0.5, random_state=42).reset_index(drop=True)
    result = pd.merge(unique_precursors, precursor_df, how="left", on=["sequence", "charge", "mods", "mod_sites"])
    result.to_csv("train_data/scaling_law_P4.csv", index=False)
    print(result.shape)

    precursor_df = precursor_df[precursor_df["mods"] == ""]
    precursor_df[["sequence", "charge"]].drop_duplicates(subset=["sequence", "charge"], inplace=True)
    precursor_df.to_csv("tests/train_data_precursor_v2.csv", index=False)
    
    # precursor_df.to_csv("total_train_data_precursor_df.csv", index=False)
    # fragment_intensity_df.to_csv("total_train_data_fragment_intensity_df.csv", index=False)
    precursor_df = pd.read_csv("train_data/scaling_law_P4.csv")
    precursor_df.fillna("", inplace=True)
    
    trainer_cfg = dict(
        accelerator='gpu',
        devices=[6, 7]
    )
    callbacks = [
        ModelCheckpoint(
            dirpath="./DDP",
            monitor="train_Loss",
            mode="min",
            save_top_k=3,
            filename="{epoch}-{train_Loss:.4f}.pt",
            verbose=True
        )
    ]
    
    additional_cfg = dict(
        callbacks=callbacks,
        max_epochs=100,
        enable_checkpointing=True,
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        check_val_every_n_epoch=None
    )
    trainer_cfg.update(additional_cfg)
    trainer = pl.Trainer(**trainer_cfg)  # lighting trainer
    
    loader = DataModule(precursor_df, fragment_intensity_df)
    loader.setup()
    del precursor_df
    del fragment_intensity_df
    gc.collect()
    
    pdeep = MSNetMS2Model()
    pdeep.charged_frag_types = ['b_z1', 'b_z2', 'y_z1', 'y_z2']
    pdeep.build(
        ModelMS2Bert,
        num_frag_types=4,
        dropout=0.1,
        nlayers=4
    )
    print("p1 parameters: {0}".format(sum(p.numel() for p in pdeep.model.parameters())))
    
    trainer.fit(
        pdeep,
        loader.train_dataloader()
    )


def check_data():
    IPX0001804001_df = pd.read_csv('PXD008722_precursor_df.csv')
    print(IPX0001804001_df["nce"].dtype)
    print(IPX0001804001_df[IPX0001804001_df['nce'].isna()])


if __name__ == "__main__":
    train()
