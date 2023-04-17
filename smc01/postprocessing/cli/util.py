import hydra
import logging
import os
import pathlib
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch

from ..dataframe_iterator import PandasBatchIteratorDataset
from ..dataset import ShuffleDataset
from ..util import concat_collate_fn


_logger = logging.getLogger(__name__)


def flatten_config(cfg: omegaconf.OmegaConf):
    return pd.json_normalize(omegaconf.OmegaConf.to_container(cfg), sep=".").to_dict(
        orient="records"
    )[0]


def make_mlflow_logger(mlflow_cfg) -> pl.loggers.mlflow.MLFlowLogger:
    tags = {"cwd": os.getcwd(), "slurm_job_id": os.getenv("SLURM_JOBID", "")}
    if "tags" in mlflow_cfg:
        tags.update(mlflow_cfg.tags)

    return pl.loggers.mlflow.MLFlowLogger(
        experiment_name=mlflow_cfg.experiment_name,
        run_name=mlflow_cfg.run_name,
        tracking_uri=mlflow_cfg.tracking_uri,
        tags=tags,
    )


def make_dataloader(cfg, dataset, shuffle=True, concat_collate=True):
    # If the dataset is an Iterable, we cannot ask the DataLoader to shuffle it, we
    # must shuffle it ourselves.
    shuffle = shuffle and not isinstance(dataset, torch.utils.data.IterableDataset)

    collate_fn = concat_collate_fn if concat_collate else None

    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=cfg.experiment.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
    )


def make_datasets(cfg):
    train_dataset = hydra.utils.instantiate(
        cfg.experiment.dataset,
        begin=cfg.experiment.split.train_begin,
        end=cfg.experiment.split.train_end,
    )

    val_dataset = hydra.utils.instantiate(
        cfg.experiment.dataset,
        begin=cfg.experiment.split.val_begin,
        end=cfg.experiment.split.val_end,
        station_subset=cfg.experiment.get("val_subset", "full"),
    )

    test_dataset = hydra.utils.instantiate(
        cfg.experiment.dataset,
        begin=cfg.experiment.split.test_begin,
        end=cfg.experiment.split.test_end,
        station_subset=cfg.experiment.get("val_subset", "full"),
    )

    n_stations = len(train_dataset.stations)
    sample = next(iter(train_dataset))
    n_features = sample["features"].shape[1]

    if "limit_dataframe_size" in cfg.experiment:
        transform = train_dataset.transform
        train_dataset.transform = None
        train_dataset = PandasBatchIteratorDataset(
            train_dataset, cfg.experiment.limit_dataframe_size, transform=transform
        )

        train_dataset = ShuffleDataset(
            train_dataset, cfg.experiment.shuffle_buffer_size
        )

        transform = val_dataset.transform
        val_dataset.transform = None
        val_dataset = PandasBatchIteratorDataset(
            val_dataset, cfg.experiment.limit_dataframe_size, transform=transform
        )

        transform = test_dataset.transform
        test_dataset.transform = None
        test_dataset = PandasBatchIteratorDataset(
            test_dataset, cfg.experiment.limit_dataframe_size, transform=transform
        )

    return train_dataset, val_dataset, test_dataset, n_stations, n_features


def find_checkpoint_file(checkpoint_path):
    checkpoint_path = pathlib.Path(checkpoint_path)

    if checkpoint_path.is_file():
        return checkpoint_path
    else:
        checkpoint_files = sorted(list(checkpoint_path.rglob("*.ckpt")))
        return checkpoint_files[-1]


class LogHyperparametersCallback(pl.Callback):
    """Callback that logs hyperparameters on training begin. This is better than
    logging the hyperparameters directly ourselves, because if the validation sanity
    check fails, nothing will be logged, and our logged outputs will be cleaner."""

    def __init__(self, hyperparameters):
        self.hparams = hyperparameters

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module.logger.log_hyperparams(flatten_config(self.hparams))
