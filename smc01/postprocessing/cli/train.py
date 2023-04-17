import logging

import hydra
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from ..lightning import SMC01Module, FreezeReferenceEmbeddings
from .util import (
    find_checkpoint_file,
    make_datasets,
    make_dataloader,
    make_mlflow_logger,
    LogHyperparametersCallback,
)

_logger = logging.getLogger(__name__)


def make_tensorboard_logger(tensorboard_cfg):
    return pl.loggers.tensorboard.TensorBoardLogger(
        save_dir="tensorboard",
        name=tensorboard_cfg.name,
        default_hp_metric=False,
    )


@hydra.main(config_path="../conf", config_name="train", version_base=None)
def cli(cfg):
    train_dataset, val_dataset, test_dataset, _, n_features = make_datasets(cfg)
    n_stations = cfg.experiment.n_stations

    _logger.info(f"Working directory: {os.getcwd()}")
    _logger.info(f"Will train using {n_features} features.")

    train_loader = make_dataloader(
        cfg, train_dataset, concat_collate=cfg.experiment.concat_collate
    )
    val_loader = make_dataloader(
        cfg, val_dataset, shuffle=False, concat_collate=cfg.experiment.concat_collate
    )
    test_loader = make_dataloader(
        cfg, test_dataset, shuffle=False, concat_collate=cfg.experiment.concat_collate
    )

    model = hydra.utils.instantiate(cfg.experiment.model, n_stations, n_features)
    model = model.to(torch.float)

    optimizer = hydra.utils.instantiate(cfg.experiment.optimizer, model.parameters())
    scheduler = hydra.utils.instantiate(cfg.experiment.scheduler, optimizer)

    if (
        "checkpoint_path" in cfg.experiment
        and cfg.experiment.checkpoint_path is not None
    ):
        module = SMC01Module.load_from_checkpoint(
            find_checkpoint_file(cfg.experiment.checkpoint_path),
            model=model,
            optimizer=optimizer,
            val_subset=cfg.experiment.get("val_subset", "full"),
            scheduler=scheduler,
            scheduler_interval=cfg.experiment.get("scheduler_interval", "epoch"),
        )
    else:
        module = SMC01Module(
            model,
            optimizer,
            val_subset=cfg.experiment.get("val_subset", "full"),
            scheduler=scheduler,
            scheduler_interval=cfg.experiment.get("scheduler_interval", "epoch"),
        )
    loggers = []

    if "mlflow" in cfg.logging:
        mlflow_logger = make_mlflow_logger(cfg.logging.mlflow)
        loggers.append(mlflow_logger)

    if "tensorboard" in cfg.logging:
        tensorboard_logger = make_tensorboard_logger(cfg.logging.tensorboard)
        loggers.append(tensorboard_logger)

    # Integrate callbacks from the configuration if any.
    if "callbacks" in cfg.experiment:
        callbacks = [hydra.utils.instantiate(c) for c in cfg.experiment.callbacks]
    else:
        callbacks = []

    # Add the default callbacks: one to always log hyperparameters and and another to
    # always log the minimum value of the loss.
    hparam_callback = LogHyperparametersCallback(cfg.experiment)
    callbacks.append(hparam_callback)

    checkpoint_callback = ModelCheckpoint(monitor="Val/Loss")
    callbacks.append(checkpoint_callback)

    lr_monitor_callback = LearningRateMonitor(
        logging_interval=cfg.experiment.get("scheduler_interval", "epoch")
    )
    callbacks.append(lr_monitor_callback)

    if "freeze_reference_set" in cfg.experiment and cfg.experiment.freeze_reference_set:
        freeze_embeddings_callback = FreezeReferenceEmbeddings()
        callbacks.append(freeze_embeddings_callback)

    if cfg.experiment.get("freeze_upper", False):
        module.model.freeze_upper()

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        limit_train_batches=cfg.experiment.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.experiment.get("limit_val_batches", 1.0),
        max_epochs=cfg.experiment.get("max_epochs", None),
        accumulate_grad_batches=cfg.experiment.get("accumulate_grad_batches", 1),
    )
    trainer.fit(module, train_loader, val_loader)
    trainer.test(module, test_loader)


if __name__ == "__main__":
    cli()
