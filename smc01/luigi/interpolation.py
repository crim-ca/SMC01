import importlib.resources
import luigi
import pandas as pd
import pathlib
import pymongo

from ..interpolate.dataset_generator import generate_dataset
from ..interpolate.forecast_reader import grib_file_to_xarray
from ..interpolate.extractor import DEFAULT_EXTRACTOR
from ..interpolate.interpolate import interpolate_at_stations
from ..interpolate.obs import MongoIEMDatabase

from .iem import FetchMetar


class InterpolateStep(luigi.Task):
    interpolated_steps_path = luigi.PathParameter()
    gdps_path = luigi.PathParameter()
    step = luigi.IntParameter()
    time = luigi.DateHourParameter()
    obs_tolerance_minutes = luigi.IntParameter(default=20)
    mongo_uri = luigi.Parameter()
    database_name = luigi.Parameter()

    def output_path(self):
        output_name = self.time.strftime("%Y%m%d%H") + f"_T{self.step:03}.parquet"
        return (
            pathlib.Path(str(self.interpolated_steps_path))
            / self.time.strftime("%Y%m%d%H")
            / output_name
        )

    def output(self):
        return luigi.LocalTarget(self.output_path())

    def requires(self):
        return [FetchMetar()]

    def run(self):
        date_string = self.time.strftime("%Y%m%d%H")
        gdps_file_name = f"CMC_glb_latlon.24x.24_{date_string}_P{self.step:03}.grib2"
        gdps_file_path = (
            pathlib.Path(str(self.gdps_path)) / date_string / gdps_file_name
        )

        if not gdps_file_path.is_file():
            # There are two naming schemes for GDPS files, try the second one.
            gdps_file_name = f"CMC_glb_{date_string}_P{self.step:03}.grib2"
            gdps_file_path = (
                pathlib.Path(str(self.gdps_path)) / date_string / gdps_file_name
            )

        step_xarray = grib_file_to_xarray(gdps_file_path, DEFAULT_EXTRACTOR)

        stations_file = importlib.resources.files("smc01").joinpath("stations.csv")
        stations_df = pd.read_csv(str(stations_file))

        interpolated_step = interpolate_at_stations(step_xarray, stations_df)

        client = pymongo.MongoClient(str(self.mongo_uri))
        database = client[str(self.database_name)]

        obs_db = MongoIEMDatabase(database)
        dataset = generate_dataset(
            interpolated_step, obs_db, self.obs_tolerance_minutes
        )

        output_path = self.output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(output_path)


class InterpolatePass(luigi.Task):
    interpolated_pass_path = luigi.PathParameter()
    time = luigi.DateHourParameter()

    def output_path(self):
        filename = self.time.strftime("%Y%m%d%H.parquet")
        return pathlib.Path(self.interpolated_pass_path) / filename

    def output(self):
        return luigi.LocalTarget(self.output_path())

    def requires(self):
        return [InterpolateStep(time=self.time, step=x) for x in range(0, 241, 3)]

    def run(self):
        input_files = sorted([x.path for x in self.input()])
        in_df = [pd.read_parquet(x) for x in input_files]

        output_path = self.output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        concatenated = pd.concat(in_df)
        concatenated.to_parquet(self.output_path())
