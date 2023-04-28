from typing import Sequence
import pandas as pd
import logging
import argparse
import os
from pathlib import Path

from utils import read_parquet_with_meta, save_parquet_with_meta

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"file:{path} does not exist")


logFormatter = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S"
)
scriptLogger = logging.getLogger("Test case builder")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
# scriptLogger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treat a database of flood, and maps them to MRIO regions")
    parser.add_argument(
        "-if",
        "--input-flooddb",
        type=file_path,
        help="The path to the flood db",
        required=True,
    )
    parser.add_argument(
        "-ir",
        "--input-repev",
        type=file_path,
        help="The path to the representative events file",
        required=True,
    )
    parser.add_argument(
        "-of",
        "--output-flooddb",
        type=str,
        help="The savepath to the flood test data",
        required=True,
    )
    parser.add_argument(
        "-or",
        "--output-repev",
        type=str,
        help="The savepath to the repev test data",
        default=1.0
    )
    args = parser.parse_args(argv)
    input_flooddb = Path(args.input_flooddb).resolve()
    input_repev = Path(args.input_repev).resolve()

    output_flooddb = Path(args.output_flooddb).resolve()
    output_repev = Path(args.output_repev).resolve()

    df = read_parquet_with_meta(input_flooddb)
    reps = read_parquet_with_meta(input_repev)

    save_parquet_with_meta(df.loc[(df["mrio_region"].isin(["AU","FR","DE","FR10", "DE11", "AUS"])) & (df.year <= 2035)], output_flooddb.parent/(output_flooddb.name+"_server.parquet"))
    save_parquet_with_meta(df.loc[(df["mrio_region"].isin(["FR","FR10"])) & (df.year <= 2035)], output_flooddb.parent/(output_flooddb.name+"_local.parquet"))

    save_parquet_with_meta(reps.loc[(reps.mrio_region.isin(["AU","FR","DE","FR10", "DE11", "AUS"]))].sort_values(by=["mrio_region","direct_prodloss_as_2010gva_share"]).groupby("mrio_region").nth([0,2,-1]).reset_index(), output_repev.parent/(output_repev.name+"_server.parquet"))
    save_parquet_with_meta(reps.loc[(reps.mrio_region.isin(["FR","FR10"]))].sort_values(by=["mrio_region","direct_prodloss_as_2010gva_share"]).groupby("mrio_region").nth([0,-1]).reset_index(),output_repev.parent/(output_repev.name+"_local.parquet") )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
