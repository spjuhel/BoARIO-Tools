from typing import Sequence
import pandas as pd
import logging
import argparse
import os
from pathlib import Path

from utils import read_parquet_with_meta, save_parquet_with_meta

def select_period(df,start,end) -> pd.DataFrame:
    return df.loc[(df.date_start.dt.year>=start) & (df.date_start.dt.year<=end)]

def filter_gva_max(df,max_gva) -> pd.DataFrame:
    return df.loc[df["share of GVA used as ARIO input"]<=max_gva]

# Find closest event to specified value
def find_neighbours(value, df, colname):
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return [upperneighbour_ind]

#find sample of event for each given class
def find_samples(df, pct_cls):
    df_copy = df.copy()
    df_copy = df_copy.reset_index()
    res = None
    percentiles = df_copy.groupby(['mrio_region'])[['direct_prodloss_as_2010gva_share']].describe(percentiles=[0.01,0.05,0.1,0.33,0.5,0.66,0.75,0.8,0.9,0.99])
    i=0
    for region in df_copy.mrio_region.unique():
        i+=1
        for cl in pct_cls:
            val = percentiles.loc[region,pd.IndexSlice[:,cl]].values[0]
            tmp = (df_copy.iloc[find_neighbours(val, df_copy.loc[(df_copy["mrio_region"]==region)], "direct_prodloss_as_2010gva_share")]).copy()
            tmp.loc[:,"class"] = cl
            if res is None:
                res = tmp.copy()
            else:
                res = pd.concat([res,tmp])
        print("Region {} done! {}/{}".format(region,i,len(df_copy.mrio_region.unique())))
    return res

def build_rep_events_from_df(df):
    ddf = find_samples(df, ['min', "1%", "5%", "10%", "33%", "50%", "66%", "75%", "80%", "90%", "99%", "max"])
    ddf = ddf.drop_duplicates(subset=["mrio_region","dmg_as_direct_prodloss (€)"]).sort_values(by=["mrio_region",'class'])
    return ddf

def build_rep_events_from_parquet(parquet,gva_max=None):
    parquet = Path(parquet)
    df = pd.read_parquet(parquet)
    ddf = build_rep_events_from_df(df)
    if gva_max is not None:
        ddf = filter_gva_max(ddf, gva_max)
    return ddf

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
scriptLogger = logging.getLogger("Rep events builder")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
# scriptLogger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treat a database of flood, and maps them to MRIO regions")
    parser.add_argument(
        "-i",
        "--input-parquet",
        type=file_path,
        help="The path to the folder with Flood data",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-parquet",
        type=str,
        help="The path to the folder to save the data",
        required=True,
    )
    parser.add_argument(
        "-G",
        "--gva-max",
        type=float,
        help="maximum share of region gva damage",
        default=1.0
    )
    args = parser.parse_args(argv)
    input_parquet = Path(args.input_parquet).resolve()
    output_parquet = Path(args.output_parquet).resolve()
    gva_max = args.gva_max

    df = build_rep_events_from_parquet(input_parquet,gva_max)
    save_parquet_with_meta(df=df,path=output_parquet)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
