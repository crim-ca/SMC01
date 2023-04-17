import hydra
import logging
import torch
import pytorch_lightning as pl

from smc01.postprocessing.lightning import SMC01Module

from .util import (
    make_dataloader,
    make_datasets,
    find_checkpoint_file,
    make_mlflow_logger,
    flatten_config,
)

_logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="validate", version_base=None)
def cli(cfg):
    _, _, test_dataset, n_stations, n_features = make_datasets(cfg)
    n_stations = cfg.experiment.n_stations

    test_loader = make_dataloader(
        cfg, test_dataset, shuffle=False, concat_collate=cfg.experiment.concat_collate
    )

    model = hydra.utils.instantiate(cfg.experiment.model, n_stations, n_features)

    module = SMC01Module(
        model=model, val_subset=cfg.experiment.get("val_subset", "full")
    )

    loggers = []

    if "mlflow" in cfg.logging:
        mlflow_logger = make_mlflow_logger(cfg.logging.mlflow)
        loggers.append(mlflow_logger)

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        limit_train_batches=cfg.experiment.get("limit_train_batches", 1.0),
        max_epochs=cfg.experiment.get("max_epochs", None),
        logger=loggers,
    )

    flat_config = flatten_config(cfg.experiment)
    for logger in loggers:
        logger.log_hyperparams(flat_config)

    if cfg.checkpoint_path is None:
        _logger.warning("Running validation without checkpoint file")
        checkpoint_path = None
    else:
        checkpoint_path = find_checkpoint_file(cfg.checkpoint_path)
    trainer.test(module, test_loader, checkpoint_path)
