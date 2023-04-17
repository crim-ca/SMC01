import collections
import pathlib

import hydra
import torch
import yaml

from .lightning import SMC01Module


def concat_collate_fn(batch):
    """Specialized collate function that concatenates the elements of the batch
    instead of stacking them. Good for tabular dataset where a batch is a DataFrame
    full of examples."""
    elem = batch[0]

    if isinstance(elem, collections.abc.Mapping):
        return {key: concat_collate_fn([b[key] for b in batch]) for key in elem}
    elif isinstance(elem, torch.Tensor):
        return torch.cat(batch, dim=0)
    else:
        raise ValueError(f"Unsupported type for concat_collate_fn: {type(elem)}.")


def find_checkpoint_file(run_dir):
    """Given a hydra run directory, find the latest checkpoint file in it and return
    its path."""
    run_path = pathlib.Path(run_dir)
    checkpoint_files = sorted(list(run_path.rglob("*.ckpt")))
    return checkpoint_files[-1]


def load_checkpoint_from_run(run_dir) -> torch.nn.Module:
    """Give the path of a hydra run, load it's checkpoint using the same overrides
    that were used to start the run."""
    overrides_file = pathlib.Path(run_dir) / ".hydra" / "overrides.yaml"
    with overrides_file.open() as f:
        overrides = yaml.safe_load(f)

    with hydra.initialize("conf"):
        cfg = hydra.compose("train", overrides)

    dataset = hydra.utils.instantiate(cfg.experiment.dataset)
    n_stations = len(dataset.stations)

    sample = dataset[0]
    n_features = sample["features"].shape[1]
    model = hydra.utils.instantiate(cfg.experiment.model, n_stations, n_features)
    optimizer = hydra.utils.instantiate(cfg.experiment.optimizer, model.parameters())

    checkpoint_file = find_checkpoint_file(run_dir)

    # We discard the full module. We only want the weights inside the model to be
    # loaded from checkpoint.
    _ = SMC01Module.load_from_checkpoint(
        checkpoint_file, model=model, optimizer=optimizer
    )

    return model
