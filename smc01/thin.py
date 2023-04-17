"""Command line utility to thin out the GRIB files we got from ECCC.
Currently, out filtering policy is very simple. We filter fields
that have very numerous vertical levels.

We use eccodes instead of pygrib because pygrib does not have write capabilities."""

import argparse

from .grib import GribFile

LEVELS_TO_KEEP = [1000, 925, 850, 700, 500, 250]


def should_keep(message, keep_levels=LEVELS_TO_KEEP):
    return message["typeOfLevel"] != "isobaricInhPa" or message["level"] in keep_levels


def cli():
    parser = argparse.ArgumentParser(
        description="Perform thinning by reading infile and writing a subset of the grib messages to outfile."
    )
    parser.add_argument("infile", type=str, help="Input file")
    parser.add_argument("outfile", type=str, help="Output file")
    args = parser.parse_args()

    with GribFile(args.infile) as infile:
        with GribFile(args.outfile, "wb") as outfile:
            for msg in infile:
                if should_keep(msg):
                    outfile.write(msg)


if __name__ == "__main__":
    cli()
