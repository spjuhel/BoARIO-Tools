from __future__ import annotations
from collections.abc import Sequence
import argparse
import json
import logging
import os
from pathlib import Path
import pickle
import re
import pandas as pd
from boario_tools.mriot import va_df_build

from utils import save_parquet_with_meta, dir_path


logFormatter = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S"
)
scriptLogger = logging.getLogger("build_gva_df")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
# scriptLogger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build gva file from mrios")
    parser.add_argument(
        "-i",
        "--input-folder",
        help="The path to the mrios folder",
        type=dir_path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The path to save the data to",
        required=True,
    )
    parser.add_argument(
        "-P",
        "--mrio-params",
        type=open,
        help="The file with shapes for the mrio",
        required=True,
    )
    parser.add_argument(
        "-E",
        "--event-params",
        type=open,
        help="The template event file",
        required=True,
    )
    args = parser.parse_args(argv)

    scriptLogger.info(
        "Make sure you use the same python environment as the one loading the pickle file (especial pymrio and pandas version !)"
    )
    scriptLogger.info(
        "Your current environment is: {} (assuming conda/mamba)".format(
            os.environ["CONDA_PREFIX"]
        )
    )
    inputf = Path(args.input_folder).resolve()
    output = Path(args.output).resolve()
    mrio_sectors_template = json.load(args.mrio_params)
    # event_template = json.load(args.event_params)
    mrio_re = re.compile(
        r"^(?P<mrio_basename>[a-zA-Z0-9]+)_(?P<year>[0-9]+)_(?P<mrio_subname>full|\d+_sectors)$"
    )
    mrio_files = list(inputf.glob("**/*.pkl"))
    print(mrio_files)
    exio_dict = {}
    for f in mrio_files:
        if match := mrio_re.match(f.stem):
            exio_dict[match["mrio_basename"]] = {}

    print(exio_dict)

    for f in mrio_files:
        if match := mrio_re.match(f.stem):
            exio_dict[match["mrio_basename"]][match["mrio_subname"]] = {}

    print(exio_dict)

    for f in mrio_files:
        if match := mrio_re.match(f.stem):
            with open(f, "rb") as mr:
                mrio = pickle.load(mr)
                exio_dict[match["mrio_basename"]][match["mrio_subname"]][
                    match["year"]
                ] = mrio

    print(exio_dict)

    for mrio_base in exio_dict:
        for mrio_sub in exio_dict[mrio_base]:
            va_df = va_df_build(
                exio_dict[mrio_base][mrio_sub].items(),
                mrio_sectors_template,
                mrio_unit=mrio_sectors_template["monetary_unit"],
            )
            save_parquet_with_meta(
                va_df, output / f"GVA_KSTOCK_GrossOutput_{mrio_base}_{mrio_sub}.parquet"
            )
            gva_df = va_df.loc[:, pd.IndexSlice[:, "GVA (€)"]].groupby("region").sum()
            gva_df = gva_df.droplevel(1, axis=1)
            save_parquet_with_meta(
                gva_df, output / f"GVA_{mrio_base}_{mrio_sub}.parquet"
            )
            final_demand_df = (
                va_df.loc[:, pd.IndexSlice[:, "yearly total final demand (€)"]]
                .groupby("region")
                .sum()
            )
            final_demand_df = final_demand_df.droplevel(1, axis=1)
            save_parquet_with_meta(
                final_demand_df, output / f"FINAL_DEMAND_{mrio_base}_{mrio_sub}.parquet"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
