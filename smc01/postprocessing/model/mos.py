import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


_logger = logging.getLogger(__name__)


class YearlyEMOS(nn.Module):
    """Yearly EMOS model. This EMOS model does not have separate weights for every
    forecast of the year. It uses the same weights to correct the forecasts from
    every day."""

    def __init__(self, n_stations, in_features, n_forecast_daily=2, n_steps_daily=8):
        super().__init__()

        self.weights = nn.Parameter(
            torch.zeros(n_stations, n_forecast_daily, n_steps_daily, in_features)
        )
        self.biases = nn.Parameter(
            torch.zeros(n_stations, n_forecast_daily, n_steps_daily, 1)
        )

        with torch.no_grad():
            self.weights[..., 0] = 1.0

        self.n_forecast_daily = n_forecast_daily
        self.n_steps_daily = n_steps_daily

    def forward(self, batch):
        features = batch["features"]

        station_ids = batch["station_id"]
        forecast_id = batch["forecast_id"] % self.n_forecast_daily
        step_id = batch["step_id"] % self.n_steps_daily

        weights = self.weights[station_ids, forecast_id, step_id]
        biases = self.biases[station_ids, forecast_id, step_id]

        pred = (weights * features).sum(dim=-1)
        pred = pred + biases.squeeze()
        return pred


class EMOS(nn.Module):
    """EMOS model that was developed for the data from the GDPS model. It makes separate
    weights for every forecast, for every step, for every day of the year.

    This EMOS model offers the option to behave as a rolling bank of linear models. If
    forecast_filter_size is > 1, then the model will use neighboring models to create
    an ensemble of models to make a single prediction. Ex: if forecast_filter_size
    is 3, and we want to make a forecast for forecast_id 10, then models 9 10 11 will
    be used and we return the average.

    Args:
        n_station: N of stations in dataset.
        in_features: N of features in the input dataset.
        n_forecast: N of forecast in a year
        n_steps: N of steps in a forecast
        forecast_filter_size: N of neighboring linear models used to make a forecast.
        forecast_filter_dilation: Use one in every X linear model to make a forecast.
        lead_filter_size: N of neighboring linear models used to make a forecast.
        lead_filter_dilation: Use one in every X linear models to make a forecast."""

    def __init__(
        self,
        n_stations,
        in_features,
        n_forecast=730,
        n_steps=81,
        forecast_filter_size=1,
        forecast_filter_dilation=1,
        lead_filter_size=1,
        lead_filter_dilation=1,
    ):
        super().__init__()

        self.weights = nn.Parameter(
            torch.zeros(n_stations, n_forecast, n_steps, in_features)
        )
        self.biases = nn.Parameter(torch.zeros(n_stations, n_forecast, n_steps, 1))

        with torch.no_grad():
            self.weights[..., 0] = 1.0

        self.forecast_filter_size = forecast_filter_size
        self.forecast_filter_dilation = forecast_filter_dilation

        self.lead_filter_size = lead_filter_size
        self.lead_filter_dilation = lead_filter_dilation

        self.n_forecast = n_forecast
        self.n_step = n_steps

    def forward(self, batch):
        features = batch["features"]

        station_ids = batch["station_id"]
        forecast_id = batch["forecast_id"]
        step_id = batch["step_id"]

        # We want indexing tensors of three dimensions: one dimension for the
        # station id, one dimension for all the forecasts id, and one dimension
        # for all the step id.

        station_selector = (
            batch["station_id"]
            .unsqueeze(1)
            .unsqueeze(2)
            .expand((-1, self.forecast_filter_size, self.lead_filter_size))
        )

        forecast_selector = torch.arange(
            0,
            self.forecast_filter_size * self.forecast_filter_dilation,
            step=self.forecast_filter_dilation,
            dtype=int,
            device=features.device,
        ).repeat(len(station_ids), 1)

        forecast_selector = (
            forecast_selector
            + forecast_id.unsqueeze(-1)
            - ((self.forecast_filter_size * self.forecast_filter_dilation) // 2)
        )

        # Fix the indices where the forecast id is negative or too large.
        # Here our strategy is to have a rolling window (so forecast id 367 maps back
        # to forecast id 2).
        forecast_selector = forecast_selector % self.n_forecast
        forecast_selector = forecast_selector.unsqueeze(-1).expand(
            -1, -1, self.lead_filter_size
        )

        step_selector = torch.arange(
            0,
            self.lead_filter_size * self.lead_filter_dilation,
            step=self.lead_filter_dilation,
            dtype=int,
            device=features.device,
        ).repeat((len(forecast_id), 1))
        step_selector = (
            step_selector
            + step_id.unsqueeze(-1)
            - ((self.lead_filter_size * self.lead_filter_dilation) // 2)
        )

        # Fix the indices where step id is negative or too large.
        # Here our strategy is to reuse the target step itself.
        invalid_step_mask = (step_selector < 0) | (step_selector >= self.n_step)
        step_selector[invalid_step_mask] = step_id.unsqueeze(1).expand(
            -1, step_selector.shape[1]
        )[invalid_step_mask]
        step_selector = step_selector.unsqueeze(1).expand(
            -1, self.forecast_filter_size, -1
        )

        gathered_weights = self.weights[
            station_selector, forecast_selector, step_selector
        ]
        gathered_biases = self.biases[
            station_selector, forecast_selector, step_selector
        ]

        features = features.unsqueeze(1).unsqueeze(2)

        # Here dim=1 is the dimension of the set of EMOS models.
        # We make the average over that dimension to model an ensemble of EMOS models.
        pred = (gathered_weights * features).mean(dim=[1, 2]).sum(dim=1)
        pred = pred + gathered_biases.mean(dim=[1, 2]).squeeze()

        return pred

    def smooth_weights(self) -> None:
        _logger.debug("Smoothing weights")

        with torch.no_grad():
            self.weights.data = smooth_emos_weights(
                self.weights.data,
                forecast_filter_size=self.forecast_filter_size,
                forecast_dilation=self.forecast_filter_dilation,
                lead_filter_size=self.lead_filter_size,
                lead_dilation=self.lead_filter_dilation,
            )
            self.biases.data = smooth_emos_weights(
                self.biases.data,
                forecast_filter_size=self.forecast_filter_size,
                forecast_dilation=self.forecast_filter_dilation,
                lead_filter_size=self.lead_filter_size,
                lead_dilation=self.lead_filter_dilation,
            )


def smooth_emos_weights(
    tensor,
    forecast_filter_size=1,
    lead_filter_size=1,
    forecast_dilation=1,
    lead_dilation=1,
):
    """Perform a smoothing operation across one dimension of a tensor. Useful
    to smooth EMOS weights across time, for instance.

    Args
        tensor: The weights to smooth.
        forecast_filter_size: The filter size on the forecast id axis
        lead_filter_size: The filter size on the lead time axis
        forecast_dilation: The dilation on the forecast axis
        lead_dilation: The filter dilation on the lead time axis

    Returns
        A tensor that had a smoothing filter applied."""

    # Put channels first.
    tensor = tensor.transpose(2, 3).transpose(1, 2)

    # Add circular padding
    left_padding = forecast_dilation * (forecast_filter_size // 2)
    right_padding = (
        forecast_dilation * (forecast_filter_size // 2)
        - forecast_dilation
        + forecast_dilation * (forecast_filter_size % 2)
    )

    top_padding = lead_dilation * (lead_filter_size // 2)
    bottom_padding = (
        lead_dilation * (lead_filter_size // 2)
        - lead_dilation
        + lead_dilation * (lead_filter_size % 2)
    )

    top_bottom_padded = F.pad(
        tensor, [top_padding, bottom_padding, 0, 0], mode="replicate"
    )
    padded_tensor = F.pad(
        top_bottom_padded, [0, 0, left_padding, right_padding], mode="circular"
    )

    # Create filter.
    # We want the filter to be the average of all the filtered values.
    n_dims = padded_tensor.shape[1]
    fltr = torch.ones(
        n_dims,
        1,
        forecast_filter_size,
        lead_filter_size,
        requires_grad=False,
        device=tensor.device,
        layout=tensor.layout,
    ) / (forecast_filter_size * lead_filter_size)

    # Perform convolution.
    # We use groups=n_dims so that one dimension at a time is filtered.
    with torch.no_grad():
        filtered = F.conv2d(
            padded_tensor,
            fltr,
            groups=n_dims,
            dilation=(forecast_dilation, lead_dilation),
        )

    filtered = filtered.transpose(1, 2).transpose(2, 3)

    return filtered
