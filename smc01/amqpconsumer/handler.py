"""A handler is a callable that accepts a message and takes action on it.
For the datamart, the messages contain the URL to files that we can download.
An example action could be to download the pointed file."""

import datetime
import pathlib
import urllib
import urllib.request


def parse_datamart_message(message):
    time, server, path = message.decode("ascii").split(" ")

    return {
        "time": parse_datamart_timestring(time),
        "server": server,
        "path": path,
    }


def parse_datamart_timestring(timestring):
    return datetime.datetime(
        int(timestring[0:4]),
        int(timestring[4:6]),
        int(timestring[6:8]),
        hour=int(timestring[8:10]),
        minute=int(timestring[10:12]),
        second=int(timestring[12:14]),
    )


class DatamartMessageHandler:
    def __call__(self, message):
        return parse_datamart_message(message)


class DatamartDownloadHandler(DatamartMessageHandler):
    def __init__(self, downloader):
        self.downloader = downloader

    def __call__(self, message):
        datamart_msg = super().__call__(message)

        target_file = pathlib.Path(datamart_msg["path"]).name
        self.downloader.download(
            datamart_msg["server"], datamart_msg["path"], target_file
        )


class GDPSDownloadHandler(DatamartMessageHandler):
    def __init__(self, downloader):
        self.downloader = downloader

    def __call__(self, message):
        datamart_msg = super().__call__(message)

        parsed_name = pathlib.Path(datamart_msg["path"]).stem.split("_")
        forecast_date_string = parsed_name[6]

        target_dir = pathlib.Path(forecast_date_string)
        target_file = target_dir / pathlib.Path(datamart_msg["path"]).name

        self.downloader.download(
            datamart_msg["server"], datamart_msg["path"], target_file
        )


class SimpleDownloader:
    def __init__(self, download_dir):
        self.download_dir = pathlib.Path(download_dir)

    def download(self, server, path, target_file):
        target_url = urllib.parse.urljoin(server, path)
        target_file = self.download_dir / pathlib.Path(target_file)

        target_file.parent.mkdir(exist_ok=True)

        urllib.request.urlretrieve(target_url, target_file)


class PrintHandler:
    def __call__(self, message):
        message = parse_datamart_message(message)


class CompositeHandler:
    def __init__(self, handlers):
        self.handlers = handlers

    def __call__(self, message):
        for h in self.handlers:
            h(message)
