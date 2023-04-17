"""Observation database reader for generating a statistical post-processing dataset."""

import collections.abc
import datetime

import pandas as pd


class MongoIEMDatabase:
    def __init__(self, db):
        self.db = db

    def _station_collection(self, station):
        return f"stn_{station}"

    def station_info(self, stations=None):
        """Return a pandas DataFrame of all stations with their coordinates.
        Args:
            station: If not none, use this list of station names to fetch station infos.
            Otherwise, use all the stations in the database."""

        if stations is None:
            stations = self.station_list()

        if isinstance(stations, collections.abc.Iterable):
            station_info = []
            for station_name in stations:
                station_info.append(self.one_station_info(station_name))
            return pd.concat(station_info).reset_index(drop=True)
        else:
            return self.one_station_info(stations)

    def one_station_info(self, station):
        collection = self.db[self._station_collection(station)]

        station_info = []
        one_obs = collection.find_one({"station": station})

        station_info.append(
            {
                "station": station,
                "lat": one_obs["lat"],
                "lon": one_obs["lon"],
                "elevation": one_obs["elevation"],
            }
        )

        return pd.DataFrame(station_info)

    def station_list(self):
        collection_names = self.db.list_collection_names()
        return [x.split("_")[-1] for x in collection_names]

    def pipeline_of_station(self, station_name, begin_date, end_date, tolerance=0):
        """Return the pertinent observations from a given station between two dates."""
        return [
            {
                "$addFields": {
                    "minute": {"$minute": "$valid"},
                    "hour": {"$hour": "$valid"},
                }
            },
            {
                "$match": {
                    "$or": [
                        {
                            "minute": {"$lte": tolerance},
                            "hour": {"$in": [0, 3, 6, 9, 12, 15, 18, 21]},
                        },
                        {
                            "minute": {"$gte": 60 - tolerance},
                            "hour": {"$in": [23, 2, 5, 8, 11, 14, 17, 20]},
                        },
                    ],
                    "station": station_name,
                    "valid": {
                        "$gte": begin_date - datetime.timedelta(minutes=tolerance),
                        "$lt": end_date + datetime.timedelta(minutes=tolerance),
                    },
                    "tmpf": {"$exists": True,},
                }
            },
        ]

    def station_observations(self, station, begin_date, end_date, tolerance=15):
        collection = self.db[self._station_collection(station)]
        pipeline = self.pipeline_of_station(
            station, begin_date, end_date, tolerance=tolerance
        )
        return list(collection.aggregate(pipeline))
