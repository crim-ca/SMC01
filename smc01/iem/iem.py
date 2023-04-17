"""Download data from the IEM archive programatically. The IEM archive contains
past METAR observations over the world.

https://mesonet.agron.iastate.edu/request/download.phtml"""

import csv
import datetime
import io
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Number of attempts to download data
MAX_ATTEMPTS = 5
# HTTPS here can be problematic for installs that don't have Lets Encrypt CA
SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
BASE = "https://mesonet.agron.iastate.edu/"
GEOJSON_SERVICE = BASE + "geojson/network/"


_logger = logging.getLogger(__name__)


def download_data(uri):
    """Fetch the data from the IEM
    The IEM download service has some protections in place to keep the number
    of inbound requests in check.  This function implements an exponential
    backoff to keep individual downloads from erroring.
    Args:
      uri (string): URL to fetch
    Returns:
      string data
    """

    retry_strategy = Retry(
        total=MAX_ATTEMPTS, status_forcelist=[429, 503], backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    r = s.get(uri)

    return r


def fetch_one_station_raw(station, begin, end):
    """Run a query to fetch the station data. Return the textual unprocessed
    response."""
    url_start = begin.strftime("year1=%Y&month1=%-m&day1=%-d")
    url_end = end.strftime("year2=%Y&month2=%-m&day2=%-d")

    options = {
        "data": "all",
        "tz": "Etc%2FUTC",
        "format": "onlycomma",
        "latlon": "yes",
        "elev": "yes",
    }

    options["station"] = station
    options_string = "&".join(["=".join(pair) for pair in options.items()])
    uri = SERVICE + "&".join([options_string, url_start, url_end])

    response = download_data(uri)
    response.raise_for_status()

    return response.text


def fetch_one_station(station, begin, end):
    """Fetch station data, and process it to a list of dicts (one dict per
    observation)."""
    raw_data = fetch_one_station_raw(station, begin, end)
    rows = raw_data_to_rows(raw_data)
    station_data = [process_row(row) for row in rows]

    return station_data


def parse_field(field, value):
    """Given a column name from the raw metar data, parse its value using the
    appropriate function."""
    if field in [
        "lon",
        "lat",
        "elevation",
        "tmpf",
        "dwpf",
        "relh",
        "drct",
        "sknt",
        "p01i",
        "alti",
        "mslp",
        "vsby",
        "gust",
        "skyl1",
        "skyl2",
        "skyl3",
        "skyl4",
        "ice_accretion_1hr",
        "ice_accretion_3hr",
        "ice_accretion_6hr",
        "peak_wind_gust",
        "peak_wind_drct",
        "feel",
    ]:
        return float(value)
    elif field in ["valid", "peak_wind_time"]:
        date = datetime.datetime.strptime(value, "%Y-%m-%d %H:%M")
        date = date.replace(tzinfo=datetime.timezone.utc)
        return date
    else:
        return value


def process_row(row):
    """Given a dict that represents one row of a raw response from the service,
    process the dict fields so that they are nice."""
    no_missing = {key: row[key] for key in row if row[key] != "M"}
    parsed = {key: parse_field(key, no_missing[key]) for key in no_missing}

    return parsed


def raw_data_to_rows(raw_data):
    stream = io.StringIO(raw_data)
    reader = csv.DictReader(stream)
    lines = (l for l in reader)
    return lines


def get_stations_from_filelist(filename):
    """Build a listing of stations from a simple file listing the stations.
    The file should simply have one station per line.
    """
    stations = []
    for line in open(filename):
        stations.append(line.strip())
    return stations


def us_networks():
    states = [
        "AK",
        "AL",
        "AR",
        "AZ",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "IA",
        "ID",
        "IL",
        "IN",
        "KS",
        "KY",
        "LA",
        "MA",
        "MD",
        "ME",
        "MI",
        "MN",
        "MO",
        "MS",
        "MT",
        "NC",
        "ND",
        "NE",
        "NH",
        "NJ",
        "NM",
        "NV",
        "NY",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VA",
        "VT",
        "WA",
        "WI",
        "WV",
        "WY",
    ]

    # IEM quirk to have Iowa AWOS sites in its own labeled network
    return ["AWOS"] + ["{}_ASOS".format(s) for s in states]


def ca_networks():
    regions = [
        "AB",
        "BC",
        "MB",
        "NB",
        "NF",
        "NS",
        "ON",
        "PE",
        "QC",
        "SK",
        "YT",
        "NT",
        "NU",
    ]
    return ["CA_{}_ASOS".format(region) for region in regions]


def get_stations_from_networks(networks):
    """Build a station list by using a bunch of IEM networks."""
    _logger.info(f"Getting station list from {len(networks)} netwotks.")

    stations = []
    for network in networks:
        stations.extend(get_stations_from_network(network))

    return stations


def get_stations_from_network(network):
    uri = GEOJSON_SERVICE + network + ".geojson"
    response = requests.get(uri)
    jdict = response.json()

    stations = [site["properties"]["sid"] for site in jdict["features"]]

    return stations
