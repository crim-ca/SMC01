"""From an XArray of a forecast and observations from a database, generate the rows
of a machine learning dataset."""

import datetime
import logging
import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


def generate_dataset(forecast, obs_db, tolerance=0, n_process=None):
    begin_date = forecast.valid.min().data.item()
    begin_date = datetime.datetime.utcfromtimestamp(begin_date // 1e9)

    end_date = forecast.valid.max().data.item()
    end_date = datetime.datetime.utcfromtimestamp(end_date // 1e9)

    responses = []
    for station in forecast.station:
        station = station.item()
        _logger.debug("Generating dataset for station %s", station)

        forecast_of_station = forecast.sel(station=station)
        observations = obs_db.station_observations(
            station, begin_date, end_date, tolerance=tolerance
        )

        responses.append(one_station_data(forecast_of_station, observations, tolerance))

    dataset = []
    for r in responses:
        dataset.extend(r)

    if len(dataset) == 0:
        raise RuntimeError("No matching observations found for dataset.")

    df = pd.DataFrame(dataset)
    df["step"] = df["step"].dt.total_seconds()

    return df


def one_station_data(station_forecast, observations, tolerance=0):
    """Associate station forecasts with their closest observation.

    Args:
      station_forecast: An XArray dataset that contains weather forecasts interpolated
        at the station.
      observations: The collected observations for this station.
    """
    if not observations:
        return []

    by_valid = {
        valid_time: group for valid_time, group in station_forecast.groupby("valid")
    }

    failed_matches = 0
    total_matches = 0
    station_data = []
    for time in by_valid:
        total_matches += 1
        obs = best_obs_for_time(time, observations, tolerance=tolerance)
        if obs is not None:
            station_data.extend(one_obs_data(by_valid[time], obs))
        else:
            failed_matches += 1

    if failed_matches / total_matches > 0.01:
        _logger.warning("Failed match rate: {:%}".format(failed_matches / total_matches))

    return station_data


def best_obs_for_time(time, observations, tolerance=0):
    """From a collection of observations, pick the observation that is closest to a
    given time.

    Args:
        time: The time we are targeting.
        observations: The list of observations.
        tolerance: The maximum distance between the observation and the time, in
            minutes.
    """
    deltas = np.array([np.datetime64(o["valid"], "ns") - time for o in observations])

    best_obs = np.argmin(np.abs(deltas))
    min_distance = np.abs(deltas[best_obs])

    if min_distance <= np.timedelta64(tolerance, "m"):
        return observations[np.argmin(np.abs(deltas))]
    else:
        _logger.debug("No corresponding observation found for time %s", str(time))
        return None


def one_obs_data(station_forecast, obs):
    obs_data = []
    for i in range(len(station_forecast.stacked_datetime_step)):
        forecast_of_step = station_forecast.isel(stacked_datetime_step=i)

        obs_data.append(one_obs_forecast_pair(forecast_of_step, obs))

    return obs_data


def temperature_rule(forecast, obs, output):
    output["obs_2t"] = farenheit_to_celcius(obs["tmpf"])

    return output


def gdps_time_rule(forecast, output):
    date, step = forecast.stacked_datetime_step.item()

    output["forecast_time"] = date.to_pydatetime()
    output["step"] = datetime.timedelta(hours=step.total_seconds() / 3600)
    output["step_id"] = int(step.total_seconds() / (3 * 3600.0))

    return output


class ObsRule:
    def __init__(self, input_key, output_key, conversion=lambda x: x):
        self.in_key = input_key
        self.out_key = output_key
        self.conversion = conversion

    def __call__(self, obs, output):
        output[self.out_key] = self.conversion(obs[self.in_key])

        return output


class OptionalObsRule(ObsRule):
    def __call__(self, obs, output):
        if self.in_key in obs:
            return super().__call__(obs, output)
        else:
            return output


def farenheit_to_celcius(farenheit):
    return (farenheit - 32) * (5.0 / 9.0)


def kelvin_to_celcius(kelvin):
    return kelvin - 273.15


def knots_to_ms(knots):
    return knots / 1.94384


def pascal_to_hectopascal(pascal):
    return pascal / 100.0


def iem_metadata_rule(obs, output):
    metadata = {
        "station": obs["station"],
        "latitude": obs["lat"],
        "longitude": obs["lon"],
        "elevation": obs["elevation"],
        "obs_valid": obs["valid"],
    }

    return {**output, **metadata}


GDPS_CONVERSIONS = {
    "prmsl": pascal_to_hectopascal,
    "2t": kelvin_to_celcius,
    "2d": kelvin_to_celcius,
    "t_850": kelvin_to_celcius,
    "t_500": kelvin_to_celcius,
}


def gdps_rule(forecast, output):
    for key in forecast.data_vars.keys():
        conversion = GDPS_CONVERSIONS.get(key, lambda x: x)

        output[f"gdps_{key}"] = conversion(forecast[key].item())

    return output


IEM_RULES = [
    iem_metadata_rule,
    OptionalObsRule("dwpt", "obs_2d", farenheit_to_celcius),
    OptionalObsRule("sknt", "obs_10si", knots_to_ms),
    OptionalObsRule("mslp", "obs_prmsl"),
    OptionalObsRule("drct", "obs_10wdir"),
    OptionalObsRule("relh", "obs_2r"),
    OptionalObsRule("p01i", "obs_p01i"),
    ObsRule("tmpf", "obs_2t", farenheit_to_celcius),
]


def process_one_iem_obs(obs):
    """Process an IEM Observation coming from our MongoDB database. Returns a dictionary
    containing uniform observation information such as temperature, etc. all in the
    units we chose"""
    datapoint = {}
    for rule in IEM_RULES:
        datapoint = rule(obs, datapoint)

    return datapoint


def process_one_gdps_forecast(gdps_forecast):
    datapoint = gdps_rule(gdps_forecast, {})
    datapoint = gdps_time_rule(gdps_forecast, datapoint)

    return datapoint


def one_obs_forecast_pair(one_forecast, one_obs):
    """Args:
    one_forecast: XArray Dataset that contains data only for one forecast, one step.
    one_obs: An observation from the database.
    """
    obs_fields = process_one_iem_obs(one_obs)
    gdps_fields = process_one_gdps_forecast(one_forecast)

    return {**obs_fields, **gdps_fields}
