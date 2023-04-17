import torch
import torch.nn as nn


class PersistenceModel(nn.Module):
    """Compute the average bias from the past n days and use this to perform a bias
    correction on the forecast. This model has no trained weights."""

    def __init__(self, n_stations, in_features):
        super().__init__()

    def forward(self, batch):
        bias = batch["ts_gdps_2t"] - batch["ts_obs_2t"]
        bias = torch.nan_to_num(bias, 0.0)
        bias = bias.mean(dim=2)

        return batch["forecast"] - bias
