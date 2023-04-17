import datetime
import importlib.resources
import logging
import math
import pathlib

import pandas as pd
import random
import torch


_logger = logging.getLogger(__name__)


def date_of_stem(stem: str) -> pd.Timestamp:
    """Generate the corresponding date from a string."""
    year = int(stem[:4])
    month = int(stem[4:6])
    day = int(stem[6:8])
    hour = int(stem[8:10])

    return pd.to_datetime(datetime.datetime(year, month, day, hour), utc=True)


def gdps_major_version_index(forecast_time: pd.Series) -> pd.Series:
    """From the date of a forecast, generate the corresponding major revision of
    the GDPS model. Returns the index of the version, not the actual version number.
    Version 6 is index 0, Version 7 is index 1, and so forth. Earlier version are not
    supported."""
    major_version = pd.Series(index=forecast_time.index, data=3)
    major_version.loc[forecast_time < "2019-07-03"] = 0
    major_version.loc[
        (forecast_time >= "2019-07-03") & (forecast_time < "2021-12-01")
    ] = 1
    major_version.loc[forecast_time > "2021-12-01"] = 2

    return major_version


STATION_SET_FILES = {
    "full": importlib.resources.files("smc01").joinpath("stations.csv"),
    "reference": importlib.resources.files("smc01").joinpath("stations_reference.csv"),
    "bootstrap": importlib.resources.files("smc01").joinpath("stations_bootstrap.csv"),
}


class SMCParquetDataset:
    """A dataset of parquet files for the SMC project."""

    def __init__(
        self,
        dataset_dir,
        transform=None,
        begin=None,
        end=None,
        data_columns=None,
        station_subset="full",
        ensure_columns=False,
        remove_step_zero=False,
    ):
        """Args
        dataset_dir: Path to the directory that contains the parquet files.
        transform: Tansform to apply to the examples.
        begin: If specified, filter all forecasts made before that date.
        end: If specified, filter all forecasts made after that date.
        data_columns: If specified, filter the examples to keep only the specified
            columns. NB some metadata columns are included even if they are not
            speficied in this list.
        ensure_columns: Make sure that all the columns we are looking for are
            present in the example. If they aren't, drop the example.
        remove_step_zero: If True, remove step 0 from the examples."""
        dataset_dir = pathlib.Path(dataset_dir)
        self.files = sorted(list(dataset_dir.glob("*.parquet")))

        if begin:
            begin = pd.to_datetime(begin, utc=True)
            self.files = [f for f in self.files if date_of_stem(f.stem) >= begin]

        if end:
            end = pd.to_datetime(end, utc=True)
            self.files = [f for f in self.files if date_of_stem(f.stem) < end]

        self.transform = transform

        if data_columns:
            # If data_columns is specified, add metadata columns + specified columns.
            self.columns = [
                "latitude",
                "longitude",
                "elevation",
                "station",
                "date",
                "step",
            ]
            self.columns.extend(data_columns)
        else:
            self.columns = None

        station_set_file = STATION_SET_FILES[station_subset]
        stations_df = pd.read_csv(str(station_set_file))
        self.station_filter = set(stations_df["station"])

        self._stations = list(stations_df["station"])

        self.ensure_columns = ensure_columns
        self.remove_step_zero = remove_step_zero

    def __getitem__(self, idx):
        _logger.debug(f"Reading file {self.files[idx]}")
        example = pd.read_parquet(self.files[idx], columns=self.columns)

        if self.station_filter is not None:
            example = example.set_index("station")

            filter_intersection = list(set(example.index) & set(self.station_filter))

            example = example.loc[filter_intersection]
            example = example.reset_index()

        if "step" in example:
            example["step"] = example["step"].astype("timedelta64[s]")

        example["padding"] = False

        example["gdps_major_version_index"] = gdps_major_version_index(
            example["forecast_time"]
        )

        if self.ensure_columns:
            example = example.dropna(subset=self.columns, axis=0, how="any")

        if self.remove_step_zero:
            example = example[example["step_id"] > 0]

        if self.transform:
            example = self.transform(example)

        return example

    def __len__(self):
        return len(self.files)

    @property
    def stations(self):
        return self._stations


class SMCIterStepDataset(torch.utils.data.IterableDataset):
    """An iterator dataset that returns the examples step by step, instead of the whole
    forecast at once."""

    def __init__(
        self,
        smc_dataset: SMCParquetDataset,
        n_steps: int = 81,
        transform=None,
        min_rows: int = None,
    ):
        """Args:
        smc_dataset: The source SMCParquetDataset. It will be returned step by step
            by the iterator.
        n_steps: Number of steps for the model. Defaults to 81 which is the number
            of steps in the GDPS model.
        transform: The transform to apply to the example.
        min_rows: If specified, filter out all the examples which have less that the
            minimum amount of rows in them.
        """
        self.smc_dataset = smc_dataset

        self.n_steps = n_steps
        self.transform = transform

        self.min_rows = min_rows

    @property
    def stations(self):
        return self.smc_dataset.stations

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            input_dataset_it = iter(self.smc_dataset)
        else:
            # Taken from:
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
            per_worker = int(
                math.ceil(len(self.smc_dataset) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.smc_dataset))

            _logger.info(f"Worker bounds: {iter_start} to {iter_end}")

            input_dataset_it = (
                self.smc_dataset[i] for i in range(iter_start, iter_end)
            )

        for example in input_dataset_it:
            for step_id in range(self.n_steps):
                step_td = pd.to_timedelta("3H") * step_id
                filtered = example[example["step"] == step_td].copy()

                # Because there is some missing data, sometimes a timestep is
                # completely missing for a given forecast. In that case, we want
                # to drop the example.
                if self.min_rows is not None and len(filtered.index) < self.min_rows:
                    _logger.warning(
                        f"Dropping time step {step_td} because not enough rows are present in the example."
                    )
                    continue

                if self.transform:
                    filtered = self.transform(filtered)

                yield filtered


class ShuffleDataset(torch.utils.data.IterableDataset):
    """Use a buffer to be able to shuffle data from an iterable dataset.

    Taken from https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/4.
    """

    def __init__(self, dataset, buffer_size: int):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for _ in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except StopIteration:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass

    @property
    def stations(self):
        return self.dataset.stations


def make_step_dataset(
    dataset_dir,
    n_steps: int = 81,
    transform=None,
    shuffle_buffer_size: int = 1,
    min_rows=None,
    add_metadata_features=False,
    **kwargs,
) -> torch.utils.data.IterableDataset:
    """Instanciating a Step dataset is getting a little complicated, to we create
    this utility function to make it easier.

    Args:
        dataset_dir: The directory containing the parquet files.
        n_steps:  The number of steps for the weather model. Defaults to 81 which is the
            number of steps in the GPDS weather model.
        transform: The transform to apply to the examples.
        shuffle_buffer_size: The size of the buffer we accumulate to randomize the order
            of the examples. Defaults to 1 which means no shuffle.
        min_rows: If specified, filter out the examples which do not have enough rows in
            them.
        **kwargs: Forwarded to SMCParquetDataset.
    """

    smc_dataset = SMCParquetDataset(dataset_dir, **kwargs)
    step_dataset = SMCIterStepDataset(
        smc_dataset, n_steps, transform=transform, min_rows=min_rows
    )
    shuffle_dataset = ShuffleDataset(step_dataset, shuffle_buffer_size)
    return shuffle_dataset
