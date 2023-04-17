import logging
import multiprocessing
import pathlib

import numpy as np
import pandas as pd
import pygrib
import xarray as xr

_logger = logging.getLogger(__name__)


def grib_date_to_pandas(date, time):
    date, time = f"{date:08}", f"{time:04}"
    date_string = f"{date[:4]}-{date[4:6]}-{date[6:8]} {time[:2]}:{time[2:4]}"
    return pd.Timestamp(date_string)


def message_to_xarray(grib_message, lats, lons, step, datetime):
    values = np.array(grib_message.values, dtype=np.float32)
    values = np.expand_dims(values, axis=0)
    values = np.expand_dims(values, axis=0)

    da = xr.DataArray(
        values,
        dims=["datetime", "step", "latitude", "longitude"],
        coords={
            "latitude": lats,
            "longitude": lons,
            "datetime": datetime,
            "step": step,
        },
    )

    da.attrs["units"] = grib_message.units

    return da


def grib_file_to_xarray(grib_file_path, extractor) -> xr.Dataset:
    _logger.info("Processing grib file %s", grib_file_path)
    gribfile = pygrib.open(str(grib_file_path))
    dataset = grib_stream_to_xarray(gribfile, extractor)
    gribfile.close()

    return dataset


def do_one_file(grib_file_path, extractor):
    return grib_file_path, grib_file_to_xarray(grib_file_path, extractor)


def grib_stream_to_xarray(grib_file, extractor):
    sample = next(iter(grib_file))

    lats, lons = sample.latlons()
    lats, lons = lats[:, 0], lons[0]

    datetime = grib_date_to_pandas(sample.dataDate, sample.dataTime)
    datetime = [datetime]

    if sample.stepUnits == 1:
        step = [pd.Timedelta(sample.step, "h")]
    else:
        raise ValueError("Unhandled step units")

    fields = {}
    grib_file.seek(0)
    for msg in grib_file:
        label = extractor(msg)

        if label:
            data_array = message_to_xarray(msg, lats, lons, step, datetime)
            fields[label] = data_array

    xr_dataset = xr.Dataset(fields)
    xr_dataset = xr_dataset.assign_coords(valid=lambda x: x.datetime + x.step)

    return xr_dataset


def pass_to_xarray(pass_dir, extractor, processes=1, limit_per_pass=None):
    input_path = pathlib.Path(pass_dir)
    input_files = sorted(list(input_path.glob("CMC_glb_*.grib2")))

    if limit_per_pass:
        input_files = input_files[:limit_per_pass]

    with multiprocessing.Pool(processes=processes) as pool:
        datasets = pool.starmap(do_one_file, [(f, extractor) for f in input_files])
    _logger.info("Done processing individual files.")

    datasets = [d for _, d in sorted(datasets, key=lambda x: x[0])]

    # Use the merge function on the two first datasets to make sure the 0th step
    # has all the steps that the 1st step has.
    # We could use merge on the whole dataset but the RAM consumption is excessive
    # so it's better to have a small merge and then a big concat.
    step0 = datasets.pop(0)
    step1 = datasets.pop(0)
    merged_beginning = xr.merge([step0, step1])
    datasets.insert(0, merged_beginning)

    for d in datasets:
        for key, data_array in merged_beginning.data_vars.items():
            if key not in d.data_vars:
                step = d.step.item()
                _logger.warning(f"Adding missing {key} variable for timestep {step}.")
                d[key] = xr.full_like(data_array, np.nan)

    _logger.info("Concatenating datasets...")
    big_dataset = xr.concat(datasets, dim="step")
    big_dataset = big_dataset.assign_coords(valid=lambda x: x.datetime + x.step)

    return big_dataset
