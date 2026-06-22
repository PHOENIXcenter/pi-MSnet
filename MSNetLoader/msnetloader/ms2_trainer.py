import numpy as np
import torch.nn.parallel
import pandas as pd
from peptdeep.model.ms2 import  pDeepModel, ModelMS2Bert
import gc
from sklearn.metrics.pairwise import cosine_similarity
from peptdeep.utils import get_available_device
from lightning.pytorch.strategies import DDPStrategy
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
from ms2_loader import AlphaPeptDeepConverter

logger = logging.getLogger("MS2")
torch.set_printoptions(threshold=np.inf)


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
        # 手动保存检查点
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

    def get_lr_factor(self, epoch):  # epoch 返回的是batch的数量
        # print(epoch)
        lr_factor = 0.5 * (
                1 + np.cos(np.pi * epoch / self.cosine_schedule_period_iters)
        )
        if epoch <= self.warmup_iters:
            lr_factor *= epoch / self.warmup_iters
        # print(lr_factor)
        return lr_factor

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
        make_dataset = functools.partial(  # 新建函数: AnnotatedSpectrumDataset
            MSNetMS2DataSets
        )
        # 合并所有 precursor_df，并记录 batch 索引（每个 batch 中 nAA 一致）
        all_precursor_rows = []
        self.dataset_to_indices = []
        current_index = 0

        for group_df in rnd_nAA:
            nAA, df_group = _grouped[group_df]
            group_df = df_group.sample(frac=1, random_state=42)  # 再次打乱每个组内部

            num_rows = len(group_df)
            num_full_batches = num_rows // self.train_batch_size

            for i in range(num_full_batches):
                batch_df = group_df.iloc[i * self.train_batch_size: (i + 1) * self.train_batch_size]
                all_precursor_rows.append(batch_df)
                batch_indices = list(range(current_index, current_index + len(batch_df)))
                self.dataset_to_indices.append(batch_indices)
                current_index += len(batch_df)

        # 合并所有样本为一个大 dataframe，构建 dataset
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
        # print("Sample idx 0:", dataset[0])  # 看返回值结构

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

def train_model():
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