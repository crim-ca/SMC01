import functools
import pathlib

import numpy as np
import pandas as pd
import torch
import xarray as xr

from .dataset import STATION_SET_FILES

# List of features we want to use in post-processing.
# The order has some importance: the features that are assumed to be most important
# are listed first, but there is nothing scientific about that choice.
POST_PROCESSING_FEATURES = [
    "gdps_2t",
    "gdps_gh_500",
    "gdps_thick",
    "gdps_q_500",
    "gdps_u_500",
    "gdps_v_500",
    "gdps_10si",
    "gdps_al",
    "gdps_10u",
    "gdps_10v",
    "gdps_2d",
    "gdps_gh_1000",
    "gdps_gh_500",
    "gdps_gh_850",
    "gdps_prmsl",
    "gdps_q_850",
    "gdps_t_500",
    "gdps_t_850",
    "elevation",
]


class CompositeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, example):
        for t in self.transforms:
            example = t(example)

        return example


def gdps_forecast_id(forecast_time: pd.Series) -> pd.Series:
    day_of_year = forecast_time.dt.dayofyear.replace(366, 365).to_numpy(copy=True) - 1
    daily_forecast = forecast_time.dt.hour.replace({0: 0, 12: 1})

    return (day_of_year * 2 + daily_forecast).to_numpy(copy=True)


def gdps_encode_forecast_hour(forecast_hour: pd.Series) -> pd.Series:
    return (forecast_hour == 0).to_numpy(copy=True)


class DataframeToExample:
    def __init__(
        self,
        features=POST_PROCESSING_FEATURES,
        add_metadata_features=True,
    ):
        stations_file = STATION_SET_FILES["full"]
        stations_df = pd.read_csv(stations_file, usecols=["station"])

        self.station_dtype = pd.api.types.CategoricalDtype(
            categories=stations_df["station"], ordered=True
        )

        if isinstance(features, int):
            self.features = POST_PROCESSING_FEATURES[:features]
        else:
            self.features = features

        self.add_metadata_features = add_metadata_features

    def __call__(self, dataframe: pd.DataFrame):
        station_ids = (
            dataframe["station"]
            .astype(self.station_dtype)
            .cat.codes.to_numpy(copy=True)
        )

        daily_forecast_idx = gdps_forecast_id(dataframe["forecast_time"])

        if "step_id" in dataframe:
            step_id = dataframe["step_id"].to_numpy()
        else:
            step_id = (dataframe["step"] / (3600 * 3)).astype(int).to_numpy(copy=True)

        target = dataframe["obs_2t"].to_numpy()

        features = self.make_features(dataframe).astype(np.float32)

        example = {
            "forecast": dataframe["gdps_2t"].to_numpy(),
            "station_id": station_ids.astype(int),
            "forecast_id": daily_forecast_idx,
            "model_id": dataframe["gdps_major_version_index"].astype(int).to_numpy(),
            "step_id": step_id,
            "features": features,
            "obs": target,
            "padding": dataframe["padding"].to_numpy(),
            "forecast_time": dataframe["forecast_time"].astype(int).to_numpy(),
        }

        example_torch = {k: torch.tensor(example[k]) for k in example}

        return example_torch

    def make_features(self, dataframe: pd.DataFrame) -> np.array:
        features = dataframe[self.features].to_numpy()

        if self.add_metadata_features:
            forecast_dayofyear = np.expand_dims(
                dataframe["forecast_time"].dt.dayofyear / 366.0, axis=1
            )
            forecast_hour = np.expand_dims(
                gdps_encode_forecast_hour(dataframe["forecast_time"].dt.hour), axis=1
            )

            metadata_features = dataframe[
                ["latitude", "longitude", "elevation"]
            ].to_numpy()

            features = np.concatenate(
                [features, metadata_features, forecast_dayofyear, forecast_hour], axis=1
            )

        return features


GDPS_MEAN = {
    "elevation": 3.6392e02,
    "gdps_10si": 3.3827e00,
    "gdps_10u": 4.8103e-01,
    "gdps_10v": 2.2835e-01,
    "gdps_10wdir": 1.9197e02,
    "gdps_2d": 3.8937e00,
    "gdps_2r": 7.1125e01,
    "gdps_2t": 9.8198e00,
    "gdps_al": 2.0382e01,
    "gdps_gh_1000": 1.3420e02,
    "gdps_gh_500": 5.6618e03,
    "gdps_gh_850": 1.4812e03,
    "gdps_prate": 3.1372e-05,
    "gdps_prmsl": 1.0165e03,
    "gdps_q_500": 8.6652e-04,
    "gdps_q_850": 5.0483e-03,
    "gdps_t_500": -1.6717e01,
    "gdps_t_850": 6.2828e00,
    "gdps_thick": 5.5258e03,
    "gdps_u_500": 1.3445e01,
    "gdps_v_500": -5.2910e-01,
    "latitude": 4.2178e01,
    "longitude": -9.5704e01,
    "obs_10si": 3.4134e00,
    "obs_10wdir": 1.6540e02,
    "obs_2r": 6.9925e01,
    "obs_2t": 1.0565e01,
    "obs_p01i": 2.6544e-03,
    "obs_prmsl": 1.0162e03,
}

GDPS_STD = {
    "gdps_10si": 2.2737e00,
    "gdps_10u": 2.7273e00,
    "gdps_10v": 2.9773e00,
    "gdps_2d": 1.1986e01,
    "gdps_2t": 1.2986e01,
    "gdps_al": 1.2725e01,
    "gdps_gh_1000": 6.5682e01,
    "gdps_gh_500": 1.9547e02,
    "gdps_gh_850": 7.3154e01,
    "gdps_prmsl": 8.2304e00,
    "gdps_q_500": 8.2185e-04,
    "gdps_q_850": 3.8303e-03,
    "gdps_t_500": 8.3867e00,
    "gdps_t_850": 1.0911e01,
    "gdps_thick": 1.9963e02,
    "gdps_u_500": 1.1286e01,
    "gdps_v_500": 1.0798e01,
    "obs_2t": 1.2662e01,
    "obs_prmsl": 8.2395e00,
    "obs_10si": 2.6762e00,
}

METAR_STATIONS_MIN_ELEVATION = -44.0  # See notebooks/phase2/011-rescale-features.ipynb.


def normalize_relative_humidity(x):
    return x / 100.0


def normalize_wind_direction(x):
    return x / 360.0


def rescale_mean(column_name, series: pd.Series) -> pd.Series:
    return (series - GDPS_MEAN[column_name]) / GDPS_STD[column_name]


def denormalize_temperature(values):
    return (values * GDPS_STD["obs_2t"]) + GDPS_MEAN["obs_2t"]


class NormalizeDataframe:
    def __init__(self):
        rescale_columns = set(
            {
                "gdps_10si",
                "gdps_10u",
                "gdps_10v",
                "gdps_2d",
                "gdps_2t",
                "gdps_al",
                "gdps_gh_1000",
                "gdps_gh_500",
                "gdps_gh_850",
                "gdps_prmsl",
                "gdps_q_500",
                "gdps_q_850",
                "gdps_t_500",
                "gdps_t_850",
                "gdps_thick",
                "gdps_u_500",
                "gdps_v_500",
                "obs_10si",
                "obs_2t",
                "obs_prmsl",
            }
        )

        self.normalization_functions = {
            k: functools.partial(rescale_mean, k) for k in rescale_columns
        }

        # The normalization functions for the less trivial fields. The other will be
        # rescaled according to their mean and standard deviation.
        other_functions = {
            "obs_2r": normalize_relative_humidity,
            "gdps_2r": normalize_relative_humidity,
            "obs_10wdir": normalize_wind_direction,
            "gdps_10wdir": normalize_wind_direction,
            "step": lambda x: x.dt.total_seconds() / 864000.0,
            "latitude": lambda x: x / 90.0,
            "longitude": lambda x: x / 180.0,
            "elevation": lambda x: (
                np.log(x - METAR_STATIONS_MIN_ELEVATION + 1e-6) - 5.0690
            )
            / 1.5448,
            "gdps_prate": lambda x: x ** (1.0 / 3.0),
            "obs_p01i": lambda x: x ** (1.0 / 3.0),
        }

        self.normalization_functions.update(other_functions)

    def __call__(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # Add a step_id column which we don't normalize.
        dataframe["step_id"] = (
            dataframe["step"].dt.total_seconds() / (3 * 3600.0)
        ).astype(int)

        for c in self.normalization_functions:
            if c in dataframe:
                dataframe[c] = self.normalization_functions[c](dataframe[c])

        return dataframe


class DataframeToXarray:
    COORDINATE_COLUMNS = ["latitude", "longitude", "elevation"]

    def __init__(self, ensure_n_dates=None):
        self.target_n_dates = ensure_n_dates

    def __call__(self, dataframe: pd.DataFrame) -> xr.Dataset:
        dataframe["step"] = pd.to_timedelta(dataframe["step"], unit="s")

        if "obs_valid" in dataframe:
            # If there is a timezone attached to the datetime, the conversion to
            # xarray does not work. The datetime object is converted to an object
            # instead of a datetime64 as expected. We remove the timezone to make
            # sure it is converted to a datetime64 when it reaches xarray.
            dataframe["obs_valid"] = dataframe["obs_valid"].dt.tz_localize(None)

        reindexed = dataframe.set_index(["station", "date", "step"])
        metadata = reindexed[self.COORDINATE_COLUMNS].groupby("station").first()

        xr_dataset = xr.Dataset.from_dataframe(
            reindexed.drop(columns=self.COORDINATE_COLUMNS)
        )
        xr_metadata = xr.Dataset.from_dataframe(metadata)

        xr_dataset = xr_dataset.assign_coords(
            valid_time=xr_dataset.date + xr_dataset.step
        )
        xr_dataset = xr_dataset.assign_coords(xr_metadata)

        if self.target_n_dates:
            target_dates = pd.date_range(
                end=xr_dataset.date[-1].values,
                periods=self.target_n_dates,
                freq="1D",
            )

            xr_dataset = xr_dataset.reindex(date=target_dates, copy=False)

        return xr_dataset


class TimeseriesXarrayToExample:
    def __init__(self, filter_nan_stations=True):
        """Args
        filter_nan_stations: If false, the output Xarrays always have the same size.
            If true, the size of the output xarray changes, because we remove the
            stations for which we don't have an observation or a forecast.
            In the output dictionary, the `stations` key gives a binary mask that
            was used to filter the invalid stations.."""
        self.filter_nan_stations = filter_nan_stations

    def __call__(self, dataset: xr.Dataset) -> dict:
        example = {}
        target_date = dataset.date[-1].data
        obs_exists_mask = dataset.obs_valid < target_date

        gdps_variables = [k for k in dataset.data_vars.keys() if k.startswith("gdps")]

        stations_to_keep = ~(dataset.obs_2t.isel(date=-1).isnull().any(dim="step")) & ~(
            dataset.gdps_2t.isel(date=-1).isnull().any(dim="step")
        )

        if self.filter_nan_stations:
            dataset = dataset.reindex(
                station=stations_to_keep.station[stations_to_keep], copy=None
            )

        # Transform dates to string because pytorch can't collate datetime numpy arrays.
        example["forecast_date"] = str(target_date)
        example["ts_date"] = [str(x) for x in dataset.date.data]
        example["ts_obs_2t"] = dataset.obs_2t.where(obs_exists_mask)
        example["stations"] = stations_to_keep
        example["ts_gdps_2t"] = dataset.gdps_2t
        example["features"] = (
            dataset[gdps_variables]
            .to_array(name="features")
            .transpose("date", "station", "step", "variable")
        )
        example["obs"] = dataset.obs_2t.isel(date=-1)
        example["forecast"] = dataset.gdps_2t.isel(date=-1)

        for k in example:
            if k in ["forecast_date", "ts_date"]:
                pass
            else:
                example[k] = torch.from_numpy(example[k].data)

        return example


class PadToLength:
    def __init__(self, length, n_stations=1226, n_forecasts=730, n_steps=81):
        self.length = length
        self.n_stations = n_stations
        self.n_forecasts = n_forecasts
        self.n_steps = n_steps

    def __call__(self, example):
        new_example = {}
        for k in example:
            tensor = example[k]
            new_shape = (self.length, *tensor.shape[1:])

            new_tensor = torch.zeros(
                new_shape, dtype=tensor.dtype, device=tensor.device
            )
            new_tensor[: tensor.shape[0]] = tensor

            new_example[k] = new_tensor

        new_example["padding"] = torch.zeros(self.length, dtype=torch.bool)
        new_example["padding"][tensor.shape[0] :] = True

        new_example["station_id"][new_example["padding"]] = self.n_stations
        new_example["forecast_id"][new_example["padding"]] = self.n_forecasts
        new_example["step_id"][new_example["padding"]] = self.n_steps

        return new_example
