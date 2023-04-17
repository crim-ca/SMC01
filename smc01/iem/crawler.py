"""Crawl the IEM data and insert observations in a MongoDB Database"""

import logging
import multiprocessing
import pandas as pd
import pymongo

from .iem import ca_networks, fetch_one_station, get_stations_from_networks, us_networks

_logger = logging.getLogger()


def get_stations_from_file(csv_file):
    """Get a list of stations from a CSV file."""
    df = pd.read_csv(csv_file)
    stations = list(df["station"])

    return stations


def get_mongo_collection(mongo_client, database, collection_name):
    db = mongo_client[database]

    collection_names = db.list_collection_names()
    collection = db[collection_name]

    if collection_name not in collection_names:
        _logger.info(
            f"Collection {collection_name} does not exist in database. Creating it."
        )
        reset_mongo_collection(collection)

    return collection


def crawl_stations(
    stations,
    begin,
    end,
    uri,
    user,
    password,
    database,
    n_process=1,
    reset_collection=False,
):
    database_info = (uri, user, password, database)
    with multiprocessing.Pool(processes=n_process) as pool:
        pool.starmap(
            crawl_one_station,
            [
                (station, begin, end, database_info, reset_collection)
                for station in stations
            ],
        )


def crawl_one_station(station, begin, end, database_info, reset_collection=False):
    uri, user, password, database = database_info
    client = pymongo.MongoClient(
        host=uri,
        username=user,
        password=password,
        authSource="admin",
    )

    return crawl_one_station_client(
        station, begin, end, client, database, reset_collection=reset_collection
    )


def crawl_one_station_client(
    station,
    begin,
    end,
    mongo_client: pymongo.MongoClient,
    database,
    reset_collection=False,
):
    """Crawl one station observations and push them to a MongoDB client."""
    collection = get_mongo_collection(mongo_client, database, f"stn_{station}")

    if reset_collection:
        reset_mongo_collection(collection)

    return crawl_one_station_collection(station, begin, end, collection)


def crawl_one_station_collection(station, begin, end, collection):
    """Crawl one station observations and push them to a MongoDB collection."""
    obs_in_db = list(collection.find({}, {"valid": 1, "_id": 0}))
    obs_in_db = set([pd.to_datetime(d["valid"], utc=True) for d in obs_in_db])

    _logger.info(f"Downloading data from {station}.")
    iem_obs = fetch_one_station(station, begin, end)
    iem_obs = sample_station_data(iem_obs)

    iem_obs_per_date = {obs["valid"]: obs for obs in iem_obs}
    iem_obs_dates = set(iem_obs_per_date.keys())
    missing_dates = iem_obs_dates - obs_in_db

    missing_documents = [iem_obs_per_date[k] for k in missing_dates]

    if len(missing_documents) > 0:
        _logger.info(f"Writing {len(missing_documents)} for station {station}.")
        collection.insert_many(missing_documents, ordered=False)
    else:
        _logger.info(f"No obs found for station {station}")


def sample_station_data(station_data):
    df = pd.DataFrame(station_data)
    df = df[
        ~df["tmpf"].isnull()
    ]  # Presence of temperature is indicative of a non-automated message.
    df.set_index("valid", drop=False, inplace=True)
    # df = df.resample('1H').nearest()  # Keep one obs per hour,
    df = df[~df.index.duplicated(keep="first")]
    df.reset_index(drop=True, inplace=True)

    if "peak_wind_time" in df.columns:
        df[["peak_wind_time"]] = (
            df[["peak_wind_time"]]
            .astype(object)
            .where(df[["peak_wind_time"]].notnull(), None)
        )

    return df.to_dict("records")


def reset_mongo_collection(collection):
    _logger.info("Resetting collection...")

    collection.drop()
    collection.create_index("valid", unique=True)


def get_stations_list(usa=False):
    networks = ca_networks()

    if usa:
        networks.extend(us_networks())

    return get_stations_from_networks(networks)
