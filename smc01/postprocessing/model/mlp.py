import torch
import torch.nn as nn

ACTIVATION_CLS = {
    "leakyrelu": nn.LeakyReLU,
    "linear": nn.Identity(),
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    def __init__(
        self,
        n_stations,
        in_features,
        n_hidden=1,
        embedding_size=128,
        station_embedding_size=16,
        dropout=0.1,
        activation="sigmoid",
    ):
        super().__init__()

        activation_class = ACTIVATION_CLS[activation]

        self.projection = nn.Sequential(
            nn.Linear(in_features + station_embedding_size, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
            activation_class(),
        )

        self.station_embedding = nn.Parameter(
            torch.rand(n_stations, station_embedding_size)
        )

        modules = []
        for _ in range(n_hidden):
            modules.append(nn.Linear(embedding_size, embedding_size, bias=False))
            modules.append(nn.BatchNorm1d(embedding_size))
            modules.append(activation_class())
            modules.append(nn.Dropout(dropout))

        self.inner = nn.ModuleList(modules)

        self.final = nn.Linear(embedding_size, 2)

    def forward(self, batch):
        features = batch["features"]
        station_ids = batch["station_id"]

        gathered_station_embeddings = self.station_embedding[station_ids]

        network_input = torch.cat([features, gathered_station_embeddings], dim=1)

        x = self.projection(network_input)

        for b in self.inner:
            x = b(x)

        # Add the network output to the original temperature.

        x = self.final(x)

        predictions = features[:, 0] * x[:, 0] + x[:, 1]

        return predictions
