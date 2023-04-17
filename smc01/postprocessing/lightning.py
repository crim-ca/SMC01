import logging

import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch

from .transform import GDPS_MEAN, GDPS_STD
from .dataset import STATION_SET_FILES

_logger = logging.getLogger(__name__)

OBS_2T_MEAN = GDPS_MEAN["obs_2t"]
OBS_2T_STD = GDPS_STD["obs_2t"]


class SMC01Module(pl.LightningModule):
    """A Pytorch Lightning training module for the SMC01 models."""

    def __init__(
        self,
        model=None,
        optimizer=None,
        scheduler=None,
        val_subset="full",
        scheduler_interval="epoch",
    ):
        """We give default values of None to most parameters, even though the module
        cannot function without these arguments. This is to keep the module compatible
        with Lightning's load_from_checkpoint function.

        Args:
            model: The PyTorch model to train.
            optimizer: The PyTorch optimizer to use.
            scheduler: The PyTorch scheduler to use.
            val_subset: The name of the subset to use for validation. Can either be
                "full", "bootstrap" or "reference".
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval

        full_stations_df = pd.read_csv(
            str(STATION_SET_FILES["full"]), usecols=["station"]
        )

        val_station_df = pd.read_csv(
            str(STATION_SET_FILES[val_subset]), usecols=["station"]
        )

        self.val_stations_mask = torch.tensor(
            [
                *full_stations_df["station"].isin(val_station_df["station"]).to_numpy(),
                False,
            ]
        )

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        not_padding = ~batch["padding"]

        if not not_padding.any():
            return 0

        prediction = self.forward(batch)[not_padding]
        obs = batch["obs"][not_padding]

        rmse = torch.sqrt(torch.square(obs - prediction).mean())

        if torch.isnan(rmse):
            _logger.error("Loss is nan for batch.")

        self.log("Train/Loss", rmse, on_step=True, on_epoch=True)

        return rmse

    def validation_step(self, batch, batch_idx):
        station_id = batch["station_id"]
        not_padding = ~batch["padding"]

        if batch["padding"].all():
            return {}

        val_mask = self.val_stations_mask.to(station_id.device)[station_id]

        mask = not_padding & val_mask

        prediction = self.forward(batch)[mask]
        obs = batch["obs"][mask]

        rmse = torch.sqrt(torch.square(obs - prediction).mean())

        if torch.isnan(rmse):
            _logger.error("Loss is nan for batch.")

        self.log("Val/Loss", rmse, on_epoch=True, on_step=False)

        with torch.no_grad():
            prediction_rescale = prediction * OBS_2T_STD + OBS_2T_MEAN
            obs_rescale = obs * OBS_2T_STD + OBS_2T_MEAN

            rmse_rescale = torch.sqrt(
                torch.square(obs_rescale - prediction_rescale).mean()
            )
            self.log("Val/RMSE", rmse_rescale, on_epoch=True, on_step=False)

            # We also log the RMSE under the val_loss key because it's that key that we
            # use for checkpoints, early_stopping, etc.
            self.log(
                "val_loss", rmse_rescale, logger=False, on_epoch=True, on_step=False
            )

        return {}

    def test_step(self, batch, batch_idx):
        not_padding = ~batch["padding"]
        station_id = batch["station_id"]

        if batch["padding"].all():
            return {}

        val_mask = self.val_stations_mask.to(station_id.device)[station_id]

        mask = not_padding & val_mask

        prediction = self.forward(batch)[mask]
        obs = batch["obs"][mask]

        rmse = torch.sqrt(torch.square(obs - prediction).mean())

        if torch.isnan(rmse):
            _logger.error("Loss is nan for batch.")

        self.log("Test/Loss", rmse, on_epoch=True, on_step=False)

        with torch.no_grad():
            prediction_rescale = prediction * OBS_2T_STD + OBS_2T_MEAN
            obs_rescale = obs * OBS_2T_STD + OBS_2T_MEAN

            rmse_rescale = torch.sqrt(
                torch.square(obs_rescale - prediction_rescale).mean()
            )
            self.log("Test/RMSE", rmse_rescale, on_epoch=True, on_step=False)

        return {}

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
                "interval": self.scheduler_interval,
                "frequency": 1,
            },
        }


class FreezeReferenceEmbeddings(pl.Callback):
    """Callback that zeroes the gradients for certain items of an embedding. Here
    we use it to freeze some station embeddings when performing finetuning.
    """

    def __init__(self):
        super().__init__()
        station_set_file = STATION_SET_FILES['reference']
        stations_df = pd.read_csv(str(station_set_file))
        subset = set(stations_df["station"])

        all_stations = pd.read_csv(str(STATION_SET_FILES['full']))['station']

        freeze_mask = [*all_stations.isin(subset).values, True]
        self.freeze_mask = torch.tensor(freeze_mask, dtype=bool)

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: SMC01Module,
        optimizer: "pl.Optimizer",
        opt_idx: int,
    ) -> None:
        freeze_mask = torch.tensor(self.freeze_mask, dtype=bool).cuda()

        new_grad = torch.where(
            ~freeze_mask.unsqueeze(-1),
            pl_module.model.station_embedding.grad,
            torch.tensor(0.0, device=pl_module.device),
        )

        pl_module.model.station_embedding.grad = new_grad
