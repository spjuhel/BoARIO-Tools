import os
from pathlib import Path
import re

from geopandas import gpd
from boario_tools import floods, mriot
import pandas as pd
import logging
from typing import Sequence
import argparse

from ..utils import file_path, dir_path, read_parquet_with_meta

# CONSTANTS
floods.shares = pd.Series(
    [0.56, 0.16, 0.20, 0.08],
    index=["residential_dmg", "industrial_dmg", "commercial_dmg", "infrastructure_dmg"],
)
floods.K_DMG_SHARE = 0.44
floods.HIST_PERIOD_NAME = "1970_2015"
floods.PROJ_PERIOD_NAME = "2016_2130"
floods.JUNCTION_YEAR = 2016


logFormatter = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S"
)
scriptLogger = logging.getLogger("MRIO-flood-mapper")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
# scriptLogger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Treat a database of flood, and maps them to MRIO regions"
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        type=dir_path,
        help="The path to the folder with Flood data",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        help="The path to the folder to save the data",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--mrios-shape",
        type=file_path,
        help="The file with shapes for the mrio",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--mrio-name",
        type=str,
        help="The name of the mrio to use",
        required=True,
    )
    parser.add_argument(
        "-P",
        "--flopros",
        type=file_path,
        help="the path to the flopros database",
    )
    parser.add_argument(
        "-r",
        "--ref-year",
        type=int,
        help="the reference year to use for gva and gross output",
    )
    parser.add_argument(
        "-M",
        "--mrio-data-dir",
        type=dir_path,
        help="path to the boario mrio and event templates dir",
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
    folder = Path(args.input_folder).resolve()
    ref_year = args.ref_year
    output = Path(args.output_folder).resolve()
    mrio_name = args.mrio_name
    mrio_re = re.compile(
        r"^(?P<mrio_basename>[a-zA-Z0-9]+)_(?P<year>\d{4})(?:_(?P<mrio_subname>full|\d+_sectors))?$"
    )
    match = mrio_re.match(mrio_name)
    if not match:
        raise ValueError(f"{mrio_name} is not a valid mrio")
    mrio_basename = match["mrio_basename"]
    mrio_name_wo_year = match["mrio_basename"] + "_" + match["mrio_subname"]
    mrio_subname = "full" if match["mrio_subname"] is None else match["mrio_subname"]
    mrios_shapes_path = args.mrios_shape
    flopros_path = args.flopros
    mrio_data_dir = Path(args.mrio_data_dir)
    scriptLogger.info(
        f"Reading files from {folder} and building initial hist and proj files"
    )
    scriptLogger.info(f"Will save to {output}")
    if (
        not (folder / "filtered_df_hist.parquet").exists()
        or not (folder / "filtered_df_proj.parquet").exists()
    ):
        floods.parse_flood_dir_save_filtered(folder, output)

    mrios_shapes = gpd.read_file(mrios_shapes_path)
    flopros = gpd.read_file(flopros_path)
    # Hist
    df_hist_path = folder / "filtered_df_hist.parquet"
    df_proj_path = folder / "filtered_df_proj.parquet"
    scriptLogger.info(f"Will compute flood base for {mrio_name_wo_year}")
    scriptLogger.info(
        """Will save in folder: {}""".format(
            folder / "builded-data" / mrio_name_wo_year
        )
    )

    if not (
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"3_clustered_floods_hist_{mrio_basename}_{mrio_subname}.parquet"
    ).exists():
        scriptLogger.info("Starting global treatment (1) for hist")
        floods.global_treatment_until_period_change(
            df_hist_path,
            mrios_shapes,
            mrio_basename,
            mrio_name_wo_year,
            folder,
            flopros,
            floods.shares,
        )

    if not (
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"3_clustered_floods_proj_{mrio_basename}_{mrio_subname}.parquet"
    ).exists():
        scriptLogger.info("Starting global treatment (1) for proj")
        floods.global_treatment_until_period_change(
            df_proj_path,
            mrios_shapes,
            mrio_basename,
            mrio_name_wo_year,
            folder,
            flopros,
            floods.shares,
        )

    df_hist = read_parquet_with_meta(
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"3_clustered_floods_hist_{mrio_basename}_{mrio_subname}.parquet"
    )
    df_proj = read_parquet_with_meta(
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"3_clustered_floods_proj_{mrio_basename}_{mrio_subname}.parquet"
    )

    if (
        not (
            folder
            / "builded-data"
            / mrio_name_wo_year
            / f"4_clustered_floods_1970_2015_{mrio_basename}_{mrio_subname}.parquet"
        ).exists()
        or not (
            folder
            / "builded-data"
            / mrio_name_wo_year
            / f"4_clustered_floods_2016_2130_{mrio_basename}_{mrio_subname}.parquet"
        ).exists()
    ):
        scriptLogger.info("Correcting periods")
        floods.period_change(df_hist, df_proj, folder, mrio_name_wo_year)

    df_hist = read_parquet_with_meta(
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"4_clustered_floods_1970_2015_{mrio_basename}_{mrio_subname}.parquet"
    )
    df_proj = read_parquet_with_meta(
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"4_clustered_floods_2016_2130_{mrio_basename}_{mrio_subname}.parquet"
    )

    scriptLogger.info("Loading reference MRIOT")
    mrio_ref = mriot.load_mrio(
        mrio_name_wo_year + "_" + str(ref_year), pkl_filepath=mrio_data_dir / "pkls"
    )

    if not (
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"5_clustered_floods_1970_2015_{mrio_basename}_{mrio_subname}_with_prodloss.parquet"
    ).exists():
        scriptLogger.info("Starting global treatment (2) for hist")
        parquet_hist = floods.global_treatment_after_period_change(
            df_hist, mrio_name_wo_year, mrio_ref, folder, period_name=HIST_PERIOD_NAME
        )

    if not (
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"5_clustered_floods_2016_2130_{mrio_basename}_{mrio_subname}_with_prodloss.parquet"
    ).exists():
        scriptLogger.info("Starting global treatment (2) for proj")
        parquet_proj = floods.global_treatment_after_period_change(
            df_proj, mrio_name_wo_year, mrio_ref, folder, period_name=PROJ_PERIOD_NAME
        )

    df_hist = read_parquet_with_meta(
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"5_clustered_floods_1970_2015_{mrio_basename}_{mrio_subname}_with_prodloss.parquet"
    )
    df_proj = read_parquet_with_meta(
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"5_clustered_floods_2016_2130_{mrio_basename}_{mrio_subname}_with_prodloss.parquet"
    )
    # symlinking_with_meta(parquet_hist,boario_flood_data/f"full_floodbase_{mrio_basename}_{mrio_subname}_{HIST_PERIOD_NAME}.parquet")
    # symlinking_with_meta(parquet_proj,boario_flood_data/f"full_floodbase_{mrio_basename}_{mrio_subname}_{PROJ_PERIOD_NAME}.parquet")
    scriptLogger.info("Saving sub-periods")
    floods.save_subperiod(
        df_hist,
        1970,
        2015,
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"6_full_floodbase_{mrio_basename}_{mrio_subname}_1970_2015.parquet",
    )
    floods.save_subperiod(
        df_proj,
        2016,
        2130,
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"6_full_floodbase_{mrio_basename}_{mrio_subname}_2016_2130.parquet",
    )
    floods.save_subperiod(
        df_proj,
        2016,
        2035,
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"6_full_floodbase_{mrio_basename}_{mrio_subname}_2016_2035.parquet",
    )
    floods.save_subperiod(
        df_proj,
        2036,
        2050,
        folder
        / "builded-data"
        / mrio_name_wo_year
        / f"6_full_floodbase_{mrio_basename}_{mrio_subname}_2036_2050.parquet",
    )
    # symlinking_with_meta(folder/"builded-data"/mrio_name_wo_year/f"6_full_floodbase_{mrio_basename}_{mrio_subname}_1970_2015.parquet",boario_flood_data/f"full_floodbase_1970_2015.parquet")
    # symlinking_with_meta(folder/"builded-data"/mrio_name_wo_year/f"6_full_floodbase_{mrio_basename}_{mrio_subname}_2016_2035.parquet",boario_flood_data/f"full_floodbase_2016_2035.parquet")
    # symlinking_with_meta(folder/"builded-data"/mrio_name_wo_year/f"6_full_floodbase_{mrio_basename}_2036_2050.parquet",boario_flood_data/f"full_floodbase_2036_2050.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
