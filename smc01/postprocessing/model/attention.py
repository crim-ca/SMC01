import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight_key = nn.Linear(in_features, out_features, bias=False)
        self.weight_query = nn.Linear(in_features, out_features, bias=False)
        self.weight_value = nn.Linear(in_features, out_features, bias=False)

        self.n_out_features = torch.tensor(float(out_features))

    def forward(self, x):
        keys = self.weight_key(x)
        query = self.weight_query(x)
        value = self.weight_value(x)

        attention = torch.bmm(query, keys.transpose(-2, -1)) / torch.sqrt(
            self.n_out_features
        )
        attention = F.softmax(attention, dim=0)

        return torch.bmm(attention, value)


class AttentionLayer(nn.Module):
    def __init__(self, n_heads, in_features, out_features):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                AttentionHead(in_features, out_features // n_heads)
                for _ in range(n_heads)
            ]
        )

        self.linear = nn.Linear(out_features, out_features)
        self.layer_norm = nn.LayerNorm([out_features])
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        xs = []
        for head in self.heads:
            out = head(x)
            xs.append(out)

        heads_cat = torch.cat(xs, dim=-1)
        out = self.linear(heads_cat)

        return self.relu(self.layer_norm(out))


class AttentionBlock(nn.Module):
    def __init__(self, n_heads, in_features, out_features, dropout=0.0):
        super().__init__()

        self.attention_layer = AttentionLayer(n_heads, in_features, out_features)
        self.feed_forward = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.LeakyReLU(0.1),
            nn.Linear(out_features, out_features),
        )

        self.layer_norm = nn.LayerNorm([out_features])
        self.layer_norm_2 = nn.LayerNorm([out_features])

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(self.attention_layer(x))
        x = self.dropout_1(x)

        x = self.layer_norm_2(self.feed_forward(x) + x)
        x = self.dropout_2(x)
        return x


class SMCAttentionModel(nn.Module):
    """Transformer-like model that performs post-processing on the SMC dataset."""

    def __init__(
        self,
        n_stations: int,
        in_features: int,
        n_heads=1,
        embedding_size=128,
        n_blocks=3,
        dropout=0.0,
        n_models=3,
        add_meta_tokens=False,
        n_forecasts=730,
        n_steps=81,
    ):
        """Args:
        n_stations: Number of stations in the dataset.
        in_features: Number of features in each example.
        n_heads: Number of attention heads to put in each attention block.
        embedding_size: Size of the features embedding. NB the attention embedding
            size is embedding_size + station_embedding_size.
        n_blocks: Number of attention blocks.
        dropout: Dropout applied in the attention blocks.
        n_models: Number of weather models in the dataset. Defaults to 3 because
            our dataset has three GDPS major revisions (6, 7 and 8).
        add_meta_tokens: If True, add contextual tokens at the end of every
            example. One token is added for the step, another for the model version,
            and another for the forecast datetime.
        n_forecasts: Number of forecasts the weather model does in a year. Defaults
            to 730 which is the number of forecasts yearly for GDPS.
        n_steps: Number of steps in a forecast for the weather model. Defaults to
            81 which is the number of steps in one GDPS forecast.
        """
        super().__init__()

        self.add_meta_tokens = add_meta_tokens

        # Here we create one more embedding for the "padding" station.
        self.station_embedding = nn.Parameter(
            torch.rand(n_stations + 1, embedding_size)
        )

        self.embedding = nn.Linear(in_features, embedding_size, bias=True)

        self.forecast_embedding = nn.Parameter(torch.rand(n_forecasts, embedding_size))
        self.step_embedding = nn.Parameter(torch.rand(n_steps, embedding_size))
        self.model_embedding = nn.Parameter(torch.rand(n_models, embedding_size))

        self.attention_layers = nn.Sequential(
            *[
                AttentionBlock(n_heads, embedding_size, embedding_size, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )

        self.regression = nn.Linear(embedding_size, 1, bias=True)

    def forward(self, batch):
        station_id = batch["station_id"]
        features = batch["features"]
        forecast = batch["forecast"]
        step_id = batch["step_id"]
        forecast_id = batch["forecast_id"]
        model_id = batch["model_id"]

        station_embeddings = self.station_embedding[station_id]

        embedded_features = self.embedding(features)
        attention_in_features = embedded_features + station_embeddings

        # Add tokens at the end of the sequence that describe the context (forecast id,
        # step id, etc).
        if self.add_meta_tokens:
            forecast_token = self.forecast_embedding[forecast_id[:, 0]]
            model_token = self.model_embedding[model_id[:, 0]]
            step_token = self.step_embedding[step_id[:, 0]]

            attention_in_features = torch.cat(
                [
                    attention_in_features,
                    forecast_token.unsqueeze(1),
                    model_token.unsqueeze(1),
                    step_token.unsqueeze(1),
                ],
                dim=1,
            )

        annotated_features = self.attention_layers(attention_in_features)

        # Remove the metadata tokens if necessary.
        if self.add_meta_tokens:
            annotated_features = annotated_features[:, :-3]

        correction = self.regression(annotated_features)

        return forecast + correction.squeeze()

    def freeze_upper(self):
        for component in [self.attention_layers, self.regression]:
            for param in component.parameters():
                param.requires_grad = False
