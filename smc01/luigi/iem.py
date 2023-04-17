import importlib.resources
import luigi
import luigi.contrib.mongodb
import pandas as pd
import pymongo

from luigi.contrib.mongodb import MongoCollectionTarget

from ..iem.crawler import crawl_one_station_collection


class MyMongoCollectionTarget(MongoCollectionTarget):
    """Need to fix an outdated call to PyMongo"""

    def __init__(self, client, database_name, collection_name, skip_check=False):
        super().__init__(client, database_name, collection_name)
        self.skip_check = skip_check

    def read(self):
        """
        Return if the target collection exists in the database
        """
        return self._collection in self.get_index().list_collection_names()

    def exists(self):
        return self.skip_check or super().exists()


class FetchMetarOfStation(luigi.Task):
    begin = luigi.DateParameter()
    end = luigi.DateParameter()
    station_name = luigi.Parameter()
    mongo_uri = luigi.Parameter()
    database_name = luigi.Parameter()
    skip_check = luigi.BoolParameter(default=False)

    def output(self):
        client = pymongo.MongoClient(str(self.mongo_uri))
        collection_name = f"stn_{self.station_name}"
        return MyMongoCollectionTarget(
            client, self.database_name, collection_name, skip_check=self.skip_check
        )

    def run(self):
        collection = self.output().get_collection()
        collection.create_index("valid", unique=True)

        crawl_one_station_collection(
            self.station_name, self.begin, self.end, collection
        )

    def requires(self):
        return []


class FetchMetar(luigi.WrapperTask):
    def requires(self):
        station_file = importlib.resources.files("smc01").joinpath("stations.csv")
        stations_df = pd.read_csv(str(station_file))

        return [FetchMetarOfStation(station_name=s) for s in stations_df["station"]]
