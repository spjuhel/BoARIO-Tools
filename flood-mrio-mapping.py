from __future__ import annotations
import argparse
import os
from collections.abc import Sequence
import logging
from pathlib import Path
import pyarrow.feather as feather
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import re
import country_converter as coco
import reverse_geocoder as rg
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from sklearn.metrics.pairwise import euclidean_distances
from tqdm.notebook import tqdm
import json
import pyarrow
import pathlib
import geopandas as gpd

from utils import read_parquet_with_meta, save_parquet_with_meta, check_na
tqdm.pandas()

# CONSTANTS
shares = pd.Series([0.56,0.16,0.20,0.08], index=["residential_dmg","industrial_dmg","commercial_dmg","infrastructure_dmg"])
K_DMG_SHARE = 0.44
HIST_PERIOD_NAME = "1970_2015"
PROJ_PERIOD_NAME = "2016_2130"
JUNCTION_YEAR = 2016

def check_df(df:pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dataframe is not of type DataFrame")
    if "return_period" not in df.columns:
        raise ValueError("Dataframe has no 'return_period' column")
    if ("lat" not in df.columns) or ("long" not in df.columns):
        raise ValueError("Dataframe lacks either 'lat', 'long' or both column(s)")

def check_flopros(flopros:gpd.GeoDataFrame):
    if not isinstance(flopros, gpd.GeoDataFrame):
        raise ValueError("Flopros dataframe is not of type GeoDataFrame")
    if "MerL_Riv" not in flopros.columns:
        raise ValueError("Dataframe has no 'MerL_Riv' column (ie merged river flood protection layer)")

def gdfy_floods(df:pd.DataFrame, crs="epsg:4326"):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat), crs = crs)
    return gdf

def join_flopros(gdf:gpd.GeoDataFrame,flopros:gpd.GeoDataFrame):
    res = gpd.sjoin(gdf,flopros[["MerL_Riv","geometry"]], how="left",predicate="within")
    res.drop(["index_right","geometry"],axis=1,inplace=True)
    res["protected"] = res["return_period"] < res["MerL_Riv"]
    return pd.DataFrame(res)


##### USED FUNCTIONS :

## CLUSTER ON DATES
def apply_date_dbscan(df, eps=4,n_jobs=1):
    dbscan = DBSCAN(eps=eps, min_samples=1,n_jobs=n_jobs).fit(np.array(df).reshape(-1,1))
    return dbscan.labels_.tolist()

## CLUSTER ON COORDINATES :

def apply_coord_dbscan(df, n_jobs=1):
    a = df.values
    b = [[v[0],v[1]] for v in a]
    #print('b:',np.asarray(b))
    kms_per_radian = 6371.0088
    epsilon = 50 / kms_per_radian
    if len(df) > 1:
        db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine',n_jobs=n_jobs).fit(b)
        return db.labels_
    else:
        return 'u'

def get_events_in_MRIO_regions(df,mrios_shapes,mrio_name):
    """Returns a GeoDataFrame of events with assigned MRIO regions.

    The function takes a dataframe `df` of events, `mrios_shapes` which is a GeoDataFrame of MRIO regions, and `mrio_name` the name of the desired MRIO dataset. The function first performs a spatial join between the events and the MRIO regions to get the events that fall within a region.
    Any events that do not fall within a region are then assigned the nearest MRIO region using a nearest join.

    The function returns a new GeoDataFrame with columns from the input `df` and additional columns `mrio_region` and `mrio` indicating the MRIO region and dataset the event was assigned to.

    Parameters
    ----------

    df (pandas.DataFrame): DataFrame of events with columns 'long' and 'lat' indicating the event's location.
    mrios_shapes (geopandas.GeoDataFrame): GeoDataFrame of MRIO regions with a column 'mrio' indicating the name of the MRIO dataset.
    mrio_name (str): Name of the desired MRIO dataset.

    Returns
    -------

    geopandas.GeoDataFrame: A GeoDataFrame of events with assigned MRIO regions and columns from the input `df` as well as additional columns `mrio_region` and `mrio` indicating the MRIO region and dataset the event was assigned to.

    Raises
    ------

    AssertionError: If the number of events in the output GeoDataFrame is not equal to the number of events in the input `df`.

    Example:
    >>> df = pd.DataFrame({'long': [0.12, 0.13, 0.14], 'lat': [51.5, 51.6, 51.7]})
    >>> mrios_shapes = geopandas.read_file('path/to/mrios_shapes.shp')
    >>> mrio_name = 'MRIO_dataset_1'
    >>> get_events_in_MRIO_regions(df, mrios_shapes, mrio_name)
      long   lat      mrio_region       mrio
    0  0.12  51.5  Region_A_mrio1  MRIO_dataset_1
    1  0.13  51.6  Region_B_mrio1  MRIO_dataset_1
    2  0.14  51.7  Region_C_mrio1  MRIO_dataset_1
    """
    mrio_shapes_df = mrios_shapes.loc[mrios_shapes.mrio==mrio_name].copy()
    print("GDFying flood base")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat), crs = 'epsg:4326')
    print("Reprojecting")
    gdf = gdf.to_crs(crs=3857)
    mrio_shapes_df = mrio_shapes_df.to_crs(crs=3857)
    print("Spatial within join")

    groups = gdf.groupby(["long","lat"])

    gdf_eu = gdf.sjoin(mrio_shapes_df, how="left", predicate="within")
    gdf_tmp = gdf_eu[gdf_eu.mrio.isna()].copy()
    gdf_tmp = gdf.loc[gdf_tmp.index]
    print("Done")
    print(f"Found {len(gdf_tmp)} unattributed events, joining them by closest distance")
    gdf_eu = gdf_eu[~gdf_eu.mrio.isna()].copy()
    gdf_tmp = gdf_tmp.sjoin_nearest(mrio_shapes_df,how="left", max_distance=30410, distance_col="distance")
    gdf_tmp = gdf_tmp.drop_duplicates()
    print("Done, merging and returning")
    gdf_eu = pd.concat([gdf_eu,gdf_tmp], axis=0)
    cols_select = df.columns.drop("geometry").union(pd.Index(["mrio_region","mrio"]))
    gdf_eu = gdf_eu[cols_select].copy()
    #gdf_eu.drop(["index_right", "distance"],axis=1,inplace=True)
    #gdf_eu = gdf_eu.to_crs(4326)
    assert len(gdf_eu) == len(df)
    return gdf_eu

def cluster_on_dates(df):
    df["date_cluster"] = df.groupby(['model', 'mrio_region'])['dayN'].transform(apply_date_dbscan)
    return df

def cluster_on_coords(df):
    df['coord'] = list(zip(np.radians(df.lat),np.radians(df.long)))
    df['cluster'] = df.groupby(['model', 'mrio_region', 'date_cluster'])['coord'].transform(apply_coord_dbscan)
    return df

def set_cluster_name(df):
    df['final_cluster'] = df['model']+'_'+df['mrio_region']+'_d'+df['date_cluster'].astype('str')+'c'+df['cluster'].astype('str')
    return df

def add_prot(df,flopros):
    df = gdfy_floods(df)
    df = df.to_crs(crs=3857)
    df = df.sjoin(flopros[["MerL_Riv","geometry"]], how="left", predicate="within")
    df_tmp = df[df.isna().any(axis=1)].copy()
    df = df[~df.isna().any(axis=1)].copy()
    df_tmp.drop(["index_right","MerL_Riv"],axis=1,inplace=True)
    df_tmp = df_tmp.sjoin_nearest(flopros,how="left", max_distance=30000, distance_col="distance")
    df = pd.concat([df,df_tmp], axis=0)
    df.drop(["index_right","geometry", "distance"],axis=1,inplace=True)
    df["protected"] = df["return_period"] < df["MerL_Riv"]
    return df

### Make clustered filtered df

def regroup_same_cell_diff_dates(tmp_df):
    """
    regroup/cluster together cells of the same event, which have same long,lat (but flooded at a different date)

    we assume that destroyed assets in the same cell correspond to the maximum assets destroyed in that flood. (ie max of "dmg")
    -> All that needs rebuilding.
    we assume that unavaillable assets (ie destroyed assets times number of days they are flooded) are the daily mean destroyed assets times the duration (ie mean of "dmg" times duration)
    -> All that induce production losses
    """
    return tmp_df.groupby(['model','final_cluster','mrio','mrio_region','long','lat']).agg(
        return_period = pd.NamedAgg(column="return_period",aggfunc="mean"),
        return_period_std = pd.NamedAgg(column="return_period",aggfunc="std"),
        date_start=pd.NamedAgg(column="date", aggfunc='min'),
        date_end=pd.NamedAgg(column="date", aggfunc='max'),
        duration=pd.NamedAgg(column="date", aggfunc='nunique'),
        # damages per each day are the sum of damages of all flooded cells in the same cluster at the same date
        total_cell_dmg=pd.NamedAgg(column="dmg", aggfunc='max'),
        mean_cell_dmg=pd.NamedAgg(column="dmg", aggfunc='mean'),
        aff_pop=pd.NamedAgg(column="aff_pop", aggfunc='max'),
        mean_aff_pop=pd.NamedAgg(column="aff_pop", aggfunc='mean')
    )

def regroup_diff_cell_same_date_cluster(tmp_df):
    """
    regroup/cluster together cells already grouped by long,lat of the same event belonging to the same date_cluster group.

    we assume that destroyed assets in differents cells add up together and correspond to the sum of the maximum assets destroyed in each cell. (ie sum of "destroyed_assets")
    -> All that needs rebuilding for the total event.
    we assume that unavaillable assets (ie destroyed assets times number of days they are flooded) are the daily mean destroyed assets times the duration (ie mean of "dmg" times duration)
    -> All that induce production losses for the total event.
    """
    return tmp_df.groupby(['model','final_cluster','mrio','mrio_region']).agg(
        long = pd.NamedAgg(column="long", aggfunc='mean'),
        lat = pd.NamedAgg(column="lat", aggfunc='mean'),
        return_period = pd.NamedAgg(column="return_period",aggfunc="mean"),
        return_period_std = pd.NamedAgg(column="return_period",aggfunc="max"),
        date_start=pd.NamedAgg(column="date_start", aggfunc='min'),
        date_end=pd.NamedAgg(column="date_end", aggfunc='max'),
        duration=pd.NamedAgg(column="duration", aggfunc='mean'),
        # damages per each day are the sum of damages of all flooded cells in the same cluster at the same date
        total_event_dmg=pd.NamedAgg(column="total_cell_dmg", aggfunc='sum'),
        daily_event_dmg=pd.NamedAgg(column="mean_cell_dmg", aggfunc='sum'),
        total_pop=pd.NamedAgg(column="aff_pop", aggfunc='sum'),
        daily_unavailable_pop=pd.NamedAgg(column="mean_aff_pop", aggfunc='sum'),

    )

def load_flood_dir(flood_data_dir,save_path):
    print(f"Looking into {flood_data_dir}")
    sc_re = re.compile(r"(r\d)")
    files = sorted(Path(flood_data_dir).glob('**/df_flood_filtered.feather'))
    print(f"Found {len(files)} files to regroup")
    projected_files = [f for f in files if 'SSP3' not in f.parent.name]
    historic_files = [f for f in files if 'SSP3' in f.parent.name]
    models = [f.parent.name for f in files if 'SSP3' not in f.parent.name]
    models = [sc_re.search(sc).group(1) for sc in models]
    print(f"Found the following models : {models}")
    df_list_proj = []
    df_list_hist = []
    for model, f in list(zip(models,historic_files)) :
        df = feather.read_feather(f)
        #df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['model'] = model
        #df = df.dropna(how='any')
        df_list_hist.append(df)

    for model, f in list(zip(models,projected_files)) :
        df = feather.read_feather(f)
        #df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['model'] = model
        #df = df.dropna(how='any')
        df_list_proj.append(df)

    filtered_df_hist = pd.concat(df_list_hist, ignore_index=True)
    filtered_df_proj = pd.concat(df_list_proj, ignore_index=True)
    save_parquet_with_meta(filtered_df_proj,save_path/"filtered_df_proj.parquet")
    save_parquet_with_meta(filtered_df_hist,save_path/"filtered_df_hist.parquet")

def compute_sector_shares(df,shares):
    tmp = pd.concat([df["total_event_dmg"]] * len(shares), axis=1, keys=shares.index) * shares
    assert np.isclose(df["total_event_dmg"], tmp.sum(axis=1)).all()
    return tmp

def global_treatment_until_period_change(initial_parquet,mrios_shapes,mrio_name,mrio_fullname,output,flopros,name=None):
    if name is None:
        if "proj" in initial_parquet.stem:
            name = "proj"
        elif "hist" in initial_parquet.stem:
            name = "hist"
        else:
            name = "noname"
    df = read_parquet_with_meta(initial_parquet)
    df = get_events_in_MRIO_regions(df,mrios_shapes,mrio_name)
    df["aff_pop"] = df["pop"]
    df.date = pd.to_datetime(df.date)
    df = cluster_on_dates(df)
    df = cluster_on_coords(df)
    df = set_cluster_name(df)
    df = df[['final_cluster', 'model', 'mrio', 'mrio_region', 'date', 'dmg', 'aff_pop', 'long', 'lat', 'return_period']].copy()
    save_parquet_with_meta(df, output/"builded-data"/mrio_fullname/f"1_pre-clustered_floods_{name}_{mrio_fullname}_with_dmg.parquet")
    flopros = flopros.to_crs(crs=3857)
    df = add_prot(df,flopros)
    save_parquet_with_meta(df,output/"builded-data"/mrio_fullname/f"2_pre-clustered_floods_{name}_{mrio_fullname}_with_dmg_with_prot.parquet")
    check_na(df[['final_cluster', 'model', 'mrio', 'mrio_region', 'date', 'dmg', 'aff_pop', 'long', 'lat', 'return_period', "MerL_Riv"]])
    df = df[~df["protected"]]
    tmp_df = df[['model','final_cluster','mrio','mrio_region','long','lat','return_period','date','dmg','aff_pop']].copy()
    df = regroup_same_cell_diff_dates(tmp_df)
    df.reset_index(inplace=True)
    tmp_df = df[['model', 'final_cluster', 'mrio', 'mrio_region', 'long', 'lat', 'return_period', 'date_start', 'date_end', 'duration', 'total_cell_dmg', 'mean_cell_dmg', 'aff_pop', 'mean_aff_pop']].copy()
    df = regroup_diff_cell_same_date_cluster(tmp_df)
    df.reset_index(inplace=True)
    df.duration = df.duration.round(0).astype(int)
    df[["residential_dmg","industrial_dmg","commercial_dmg","infrastructure_dmg"]] = compute_sector_shares(df,shares)
    df["total_capital_dmg"] = df['total_event_dmg'] * K_DMG_SHARE
    df.rename(columns={"total_capital_dmg":"Total direct damage to capital (2010€PPP)","total_pop":"Population aff (2015 est.)"},inplace=True)
    save_parquet_with_meta(df,output/"builded-data"/mrio_fullname/f"3_clustered_floods_{name}_{mrio_fullname}.parquet")

def period_change(df_hist, df_proj,output,mrio_name):
    df_hist = pd.concat([df_hist,df_proj.loc[(df_proj.date_start.dt.year<JUNCTION_YEAR)]],axis=0)
    df_proj = df_proj.loc[(df_proj.date_start.dt.year>=JUNCTION_YEAR)]
    df_hist["final_cluster"] = df_hist["final_cluster"] + "_" + df_hist["date_start"].dt.year.astype('str')
    df_proj["final_cluster"] = df_proj["final_cluster"] + "_" + df_proj["date_start"].dt.year.astype('str')
    save_parquet_with_meta(df_hist,output/"builded-data"/mrio_name/f"4_clustered_floods_{HIST_PERIOD_NAME}_{mrio_name}.parquet")
    save_parquet_with_meta(df_proj,output/"builded-data"/mrio_name/f"4_clustered_floods_{PROJ_PERIOD_NAME}_{mrio_name}.parquet")

def direct_prodloss(df,va_df, event_template, ref_year):
    print(event_template)
    aff_2010kstock = (va_df.loc[pd.IndexSlice[:,event_template['aff_sectors']],pd.IndexSlice[:,"K_stock (€)"]]).droplevel(1,axis=1).groupby("region").sum()[str(ref_year)]
    aff_sector_prod = (va_df.loc[pd.IndexSlice[:,event_template['aff_sectors']],pd.IndexSlice[:,"yearly gross output (€)"]]).droplevel(1,axis=1).groupby("region").sum()[str(ref_year)] / 365
    def prodloss(region,delta_kapital):
        return aff_sector_prod.loc[region] * (delta_kapital / aff_2010kstock.loc[region])
    prodloss_vec = np.frompyfunc(prodloss, 2,1)
    return prodloss_vec(df['mrio_region'],df["Total direct damage to capital (2010€PPP)"])

def compute_direct_prodloss(df,gva_df,va_df,event_template,ref_year):

    def dmg_as_gva_share(dmg, region, year):
        return dmg/gva_df.loc[region,str(year)]

    def dmg_as_gva_share2010(dmg, region):
        return dmg_as_gva_share(dmg, region, ref_year)

    def get_GVA(region,year):
        return gva_df.loc[region,str(year)]

    def get_GVA2010(region):
        return gva_df.loc[region,str(ref_year)]

    dmg_as_gva_share_vec = np.frompyfunc(dmg_as_gva_share, 3,1)
    dmg_as_gva_share_refyear_vec = np.frompyfunc(dmg_as_gva_share2010, 2,1)
    get_GVA_vec = np.frompyfunc(get_GVA,2,1)
    get_GVA_refyear_vec = np.frompyfunc(get_GVA2010,1,1)

    df[['date_end','date_start']] = df[['date_end','date_start']].apply(pd.to_datetime) #if conversion required
    df['year'] = df['date_start'].dt.year
    min_year = int(va_df.columns.get_level_values(0).min())
    max_year = int(va_df.columns.get_level_values(0).max())
    df['closest_MRIO_year'] = [int(year) if min_year <= int(year) <= max_year else min_year if int(year) < min_year else max_year for year in df["year"]]

    #df['duration'] = (df['date_end'] - df['date_start']).dt.days + 1
    df['dmg_as_closest_year_gva_share'] = dmg_as_gva_share_vec(df["Total direct damage to capital (2010€PPP)"],df["mrio_region"],df["closest_MRIO_year"])
    df['dmg_as_closest_year_gva_share'] = df['dmg_as_closest_year_gva_share'].astype(float)

    df['dmg_as_2010_gva_share'] = dmg_as_gva_share_refyear_vec(df["Total direct damage to capital (2010€PPP)"],df["mrio_region"])
    df['dmg_as_2010_gva_share'] = df['dmg_as_2010_gva_share'].astype(float)
    df['share of GVA used as ARIO input'] = df['dmg_as_2010_gva_share']

    df['2010 GVA (M€)'] = get_GVA_refyear_vec(df['mrio_region'])
    df['2010 GVA (M€)'] = df['2010 GVA (M€)'].astype(float)

    df['Closest flood year GVA (M€)'] = get_GVA_vec(df['mrio_region'],df['closest_MRIO_year'])
    df['Closest flood year GVA (M€)'] = df['Closest flood year GVA (M€)'].astype(float)

    df['dmg_as_direct_prodloss (€)'] = direct_prodloss(df, va_df, event_template, ref_year)
    df['dmg_as_direct_prodloss (M€)'] = df['dmg_as_direct_prodloss (€)'] / 10**6
    df['direct_prodloss_as_2010gva_share'] = df['dmg_as_direct_prodloss (M€)'] / df['2010 GVA (M€)']
    return df

# (initial_parquet,mrios_shapes,mrio_name,output,flopros,name=None):
# exiobase3_74_sectors
def global_treatment_after_period_change(df,mrio_name,output,boario_builded_data, ref_year, name):
    mrio_re = re.compile(r"^(?P<mrio_basename>[a-zA-Z0-9]+)(?:_(?P<mrio_subname>full|\d+_sectors))?$")
    match = mrio_re.match(mrio_name)
    if not match:
        raise ValueError(f"{mrio_name} is not a valid mrio")

    mrio_basename = match['mrio_basename']
    #mrio_subname = "full" if match['mrio_subname'] is None else match["mrio_subname"]
    va_df_path = pathlib.Path(boario_builded_data)/"pkls"/f"GVA_KSTOCK_GrossOutput_{mrio_name}.parquet"
    gva_df_path = pathlib.Path(boario_builded_data)/"pkls"/f"GVA_{mrio_name}.parquet"
    event_template_path = pathlib.Path(boario_builded_data)/"params"/f"{mrio_name}_event_params_rebuilding.json"

    va_df = read_parquet_with_meta(va_df_path)
    gva_df = read_parquet_with_meta(gva_df_path)
    with event_template_path.open("r") as f:
        event_template = json.load(f)

    df = compute_direct_prodloss(df,gva_df,va_df,event_template,ref_year)
    save_parquet_with_meta(df,output/"builded-data"/mrio_name/f"5_clustered_floods_{name}_{mrio_name}_with_prodloss.parquet")
    return output/"builded-data"/mrio_name/f"5_clustered_floods_{name}_{mrio_name}_with_prodloss.parquet"

def symlinking(input_file, output_file):
    pathlib.Path(output_file).unlink(missing_ok=True)
    pathlib.Path(output_file).symlink_to(input_file)

def symlinking_with_meta(input_parquet, output_parquet):
    input_parquet = pathlib.Path(input_parquet)
    output_parquet = pathlib.Path(output_parquet)
    symlinking(input_parquet.with_name(input_parquet.stem+"_meta.json"),output_parquet.with_name(output_parquet.stem+"_meta.json"))
    symlinking(input_parquet,output_parquet)

def save_subperiod(df,start,end,path):
    save_parquet_with_meta(df.loc[(df.date_start.dt.year>=start) & (df.date_start.dt.year<=end)],path)

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
scriptLogger = logging.getLogger("MRIO-flood-mapper")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
# scriptLogger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treat a database of flood, and maps them to MRIO regions")
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
        "-s", "--mrios-shape", type=file_path, help="The file with shapes for the mrio", required=True
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
        "-B",
        "--boario-builded",
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
    mrio_re = re.compile(r"^(?P<mrio_basename>[a-zA-Z0-9]+)(?:_(?P<mrio_subname>full|\d+_sectors))?$")
    match = mrio_re.match(mrio_name)
    if not match:
        raise ValueError(f"{mrio_name} is not a valid mrio")
    mrio_basename = match['mrio_basename']
    mrio_subname = "full" if match['mrio_subname'] is None else match["mrio_subname"]
    mrios_shapes_path = args.mrios_shape
    flopros_path = args.flopros
    boario_builded_data = args.boario_builded
    scriptLogger.info(f"Reading files from {folder} and building initial hist and proj files")
    scriptLogger.info(f"Will save to {output}")
    if not (folder/"filtered_df_hist.parquet").exists() or not (folder/"filtered_df_proj.parquet").exists():
        load_flood_dir(folder,output)

    mrios_shapes = gpd.read_file(mrios_shapes_path)
    flopros = gpd.read_file(flopros_path)
    # Hist
    df_hist_path = folder/"filtered_df_hist.parquet"
    df_proj_path = folder/"filtered_df_proj.parquet"
    scriptLogger.info(f"Will compute flood base for {mrio_name}")
    scriptLogger.info("""Will save in folder: {}""".format(folder/'builded-data'/mrio_name))

    if not (folder/"builded-data"/mrio_name/f"3_clustered_floods_hist_{mrio_basename}_{mrio_subname}.parquet").exists():
        scriptLogger.info("Starting global treatment (1) for hist")
        global_treatment_until_period_change(df_hist_path,mrios_shapes,mrio_basename,mrio_name,folder,flopros)

    if not (folder/"builded-data"/mrio_name/f"3_clustered_floods_proj_{mrio_basename}_{mrio_subname}.parquet").exists():
        scriptLogger.info("Starting global treatment (1) for proj")
        global_treatment_until_period_change(df_proj_path,mrios_shapes,mrio_basename,mrio_name,folder,flopros)

    df_hist = read_parquet_with_meta(folder/"builded-data"/mrio_name/f"3_clustered_floods_hist_{mrio_basename}_{mrio_subname}.parquet")
    df_proj = read_parquet_with_meta(folder/"builded-data"/mrio_name/f"3_clustered_floods_proj_{mrio_basename}_{mrio_subname}.parquet")

    if not (folder/"builded-data"/mrio_name/f"4_clustered_floods_1970_2015_{mrio_basename}_{mrio_subname}.parquet").exists() or not (folder/"builded-data"/mrio_name/f"4_clustered_floods_2016_2130_{mrio_basename}_{mrio_subname}.parquet").exists():
        scriptLogger.info("Changing the periods")
        period_change(df_hist,df_proj,folder,mrio_name)

    df_hist = read_parquet_with_meta(folder/"builded-data"/mrio_name/f"4_clustered_floods_1970_2015_{mrio_basename}_{mrio_subname}.parquet")
    df_proj = read_parquet_with_meta(folder/"builded-data"/mrio_name/f"4_clustered_floods_2016_2130_{mrio_basename}_{mrio_subname}.parquet")

    if not (folder/"builded-data"/mrio_name/f"5_clustered_floods_1970_2015_{mrio_basename}_{mrio_subname}_with_prodloss.parquet").exists():
        scriptLogger.info("Starting global treatment (2) for hist")
        parquet_hist = global_treatment_after_period_change(df_hist,mrio_name,folder,boario_builded_data=boario_builded_data, ref_year=ref_year, name=HIST_PERIOD_NAME)

    if not (folder/"builded-data"/mrio_name/f"5_clustered_floods_2016_2130_{mrio_basename}_{mrio_subname}_with_prodloss.parquet").exists():
        scriptLogger.info("Starting global treatment (2) for proj")
        parquet_proj = global_treatment_after_period_change(df_proj,mrio_name,folder,boario_builded_data=boario_builded_data, ref_year=ref_year, name=PROJ_PERIOD_NAME)

    boario_flood_data = pathlib.Path(boario_builded_data).parent/"source-data"/"flood-data"
    #symlinking_with_meta(parquet_hist,boario_flood_data/f"full_floodbase_{mrio_basename}_{mrio_subname}_{HIST_PERIOD_NAME}.parquet")
    #symlinking_with_meta(parquet_proj,boario_flood_data/f"full_floodbase_{mrio_basename}_{mrio_subname}_{PROJ_PERIOD_NAME}.parquet")
    save_subperiod(df_hist,1970,2015,folder/"builded-data"/mrio_name/f"6_full_floodbase_{mrio_basename}_{mrio_subname}_1970_2015.parquet")
    save_subperiod(df_proj,2016,2035,folder/"builded-data"/mrio_name/f"6_full_floodbase_{mrio_basename}_{mrio_subname}_2016_2035.parquet")
    save_subperiod(df_proj,2036,2050,folder/"builded-data"/mrio_name/f"6_full_floodbase_{mrio_basename}_{mrio_subname}_2036_2050.parquet")
    #symlinking_with_meta(folder/"builded-data"/mrio_name/f"6_full_floodbase_{mrio_basename}_{mrio_subname}_1970_2015.parquet",boario_flood_data/f"full_floodbase_1970_2015.parquet")
    #symlinking_with_meta(folder/"builded-data"/mrio_name/f"6_full_floodbase_{mrio_basename}_{mrio_subname}_2016_2035.parquet",boario_flood_data/f"full_floodbase_2016_2035.parquet")
    #symlinking_with_meta(folder/"builded-data"/mrio_name/f"6_full_floodbase_{mrio_basename}_2036_2050.parquet",boario_flood_data/f"full_floodbase_2036_2050.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
