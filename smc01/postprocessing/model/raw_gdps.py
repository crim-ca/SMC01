import torch.nn as nn


class RawGDPS(nn.Module):
    """Simply return the GDPS forecast. Useful when we want to have a baseline go 
    through our training and validation pipeline."""
    def __init__(self, n_stations, in_features):
        super().__init__()

    def forward(self, batch):
        return batch["forecast"]
