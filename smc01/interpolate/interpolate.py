"""Interpolate GRIB data at stations and generate a dataset to train post-processing
algorithms. This module is intended to generate dataset chunks (one pass at a time) that
can then be merged to form a yearly dataset."""

import logging
import pathlib

import hydra
import importlib.resources
import pandas as pd
import pymongo
import xarray as xr

from .dataset_generator import generate_dataset
from .extractor import DEFAULT_EXTRACTOR
from .forecast_reader import pass_to_xarray
from .obs import MongoIEMDatabase

_logger = logging.getLogger(__name__)


def database_factory(cfg):
    client = pymongo.MongoClient(
        host=cfg.host,
        port=cfg.port,
        tz_aware=True,
        authSource="admin",
        username=cfg.username,
        password=cfg.password,
    )

    return client


def interpolate_at_stations(dataset, stations):
    """Args:
    dataset: An xarray dataset containing the weather model grids.
    stations: A Pandas DataFrame containing three columns: station, lat, lon.
    """
    _logger.info("Interpolating forecast at stations...")
    at_stations = dataset.interp(
        {
            "latitude": xr.DataArray(stations["latitude"], dims="station"),
            "longitude": xr.DataArray(stations["longitude"], dims="station"),
        }
    )

    at_stations = at_stations.assign_coords(
        station=xr.DataArray(stations["station"], dims="station")
    )
    _logger.info("Done interpolating forecast at stations.")

    return at_stations


def stations_from_db(obs_database):
    _logger.info("Reading station list from database.")
    return obs_database.station_info()


def stations_from_csv(csv_stream):
    _logger.info("Reading station list from csv.")
    return pd.read_csv(csv_stream)


def interpolate(
    input_dir,
    output_file,
    obs_database,
    station_list,
    n_workers,
    limit_per_pass=None,
    obs_tolerance=20,
):
    with database_factory(obs_database) as mongo_client:
        obs_database = MongoIEMDatabase(client=mongo_client, db=obs_database.database)

        with open(station_list) as csv_file:
            stations = stations_from_csv(csv_file)

        dataset = pass_to_xarray(
            input_dir,
            DEFAULT_EXTRACTOR,
            processes=n_workers,
            limit_per_pass=limit_per_pass,
        )

        at_stations = interpolate_at_stations(dataset, stations)
        dataset = generate_dataset(
            at_stations,
            obs_database,
            tolerance=obs_tolerance,
            n_process=n_workers,
        )

    output_file.parent.mkdir(exist_ok=True, parents=True)
    dataset.to_parquet(output_file)


@hydra.main(config_name="conf", config_path="conf", version_base=None)
def cli(cfg):
    input_path = pathlib.Path(hydra.utils.to_absolute_path(cfg.input_dir))
    output_dir = pathlib.Path(hydra.utils.to_absolute_path(cfg.output_dir))

    valid_dirs = sorted([x for x in input_path.iterdir() if len(x.name) == 10])
    if "index" in cfg and cfg.index is not None:
        input_dir = valid_dirs[cfg.index]
    else:
        print(f"There are {len(valid_dirs)} valid directories in the provided path.")

        for i, dir in enumerate(valid_dirs):
            print(f"{i}: {dir}")

        return

    output_file = output_dir / f"{input_dir.name}.{cfg.export.extension}"
    if not cfg.overwrite and output_file.is_file():
        _logger.info("Skipping task because output %s already exists.", output_file)
        return

    station_list = importlib.resources.files("smc01").joinpath("stations.csv")
    interpolate(
        input_dir,
        output_file,
        cfg.database,
        station_list,
        n_workers=cfg.processes,
        limit_per_pass=cfg.limit_per_pass,
    )


if __name__ == "__main__":
    cli()
