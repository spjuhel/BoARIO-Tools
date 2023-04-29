from __future__ import annotations
from pathlib import Path
import pyarrow.feather as feather
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.cluster import DBSCAN
import pathlib
import geopandas as gpd
import pymrio as pym

from ..utils import read_parquet_with_meta, save_parquet_with_meta, check_na


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
    #scriptLogger.info('b:',np.asarray(b))
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
    scriptLogger.info("Associating events with MRIOT regions")
    mrio_shapes_df = mrios_shapes.loc[mrios_shapes.mrio==mrio_name].copy()
    scriptLogger.info("...GDFying flood base")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat), crs = 'epsg:4326')
    scriptLogger.info("...Reprojecting")
    gdf = gdf.to_crs(crs=3857)
    mrio_shapes_df = mrio_shapes_df.to_crs(crs=3857)
    scriptLogger.info("...Spatial within join")

    groups = gdf.groupby(["long","lat"])

    gdf_eu = gdf.sjoin(mrio_shapes_df, how="left", predicate="within")
    gdf_tmp = gdf_eu[gdf_eu.mrio.isna()].copy()
    gdf_tmp = gdf.loc[gdf_tmp.index]
    scriptLogger.info("......Done")
    scriptLogger.info(f"...Found {len(gdf_tmp)} unattributed events, joining them by closest distance")
    gdf_eu = gdf_eu[~gdf_eu.mrio.isna()].copy()
    gdf_tmp = gdf_tmp.sjoin_nearest(mrio_shapes_df,how="left", max_distance=30410, distance_col="distance")
    gdf_tmp = gdf_tmp.drop_duplicates()
    scriptLogger.info("......Done, merging and returning")
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
    we assume that unavaillable assets (ie destroyed assets times number of days they are flooded) are the daily mean destroyed assets times the time_extent (ie mean of "dmg" times time_extent)
    -> All that induce production losses
    """
    return tmp_df.groupby(['model','final_cluster','mrio','mrio_region','long','lat']).agg(
        return_period = pd.NamedAgg(column="return_period",aggfunc="mean"),
        return_period_std = pd.NamedAgg(column="return_period",aggfunc="std"),
        date_start=pd.NamedAgg(column="date", aggfunc='min'),
        date_end=pd.NamedAgg(column="date", aggfunc='max'),
        time_extent=pd.NamedAgg(column="date", aggfunc='nunique'),
        # damages per each day are the sum of damages of all flooded cells in the same cluster at the same date
        total_cell_dmg=pd.NamedAgg(column="dmg", aggfunc='max'),
        aff_pop=pd.NamedAgg(column="aff_pop", aggfunc='max'),
    )

def regroup_diff_cell_same_date_cluster(tmp_df):
    """
    regroup/cluster together cells already grouped by long,lat of the same event belonging to the same date_cluster group.

    we assume that destroyed assets in differents cells add up together and correspond to the sum of the maximum assets destroyed in each cell. (ie sum of "destroyed_assets")
    -> All that needs rebuilding for the total event.
    we assume that unavaillable assets (ie destroyed assets times number of days they are flooded) are the daily mean destroyed assets times the time_extent (ie mean of "dmg" times time_extent)
    -> All that induce production losses for the total event.
    """
    return tmp_df.groupby(['model','final_cluster','mrio','mrio_region']).agg(
        long = pd.NamedAgg(column="long", aggfunc='mean'),
        lat = pd.NamedAgg(column="lat", aggfunc='mean'),
        return_period = pd.NamedAgg(column="return_period",aggfunc="mean"),
        return_period_std = pd.NamedAgg(column="return_period",aggfunc="max"),
        date_start=pd.NamedAgg(column="date_start", aggfunc='min'),
        date_end=pd.NamedAgg(column="date_end", aggfunc='max'),
        time_extent=pd.NamedAgg(column="time_extent", aggfunc='mean'),
        # damages per each day are the sum of damages of all flooded cells in the same cluster at the same date
        total_event_dmg=pd.NamedAgg(column="total_cell_dmg", aggfunc='sum'),
        total_pop=pd.NamedAgg(column="aff_pop", aggfunc='sum'),
    )

def parse_flood_dir_save_filtered(flood_data_dir,save_path):
    scriptLogger.info(f"Looking into {flood_data_dir}")
    sc_re = re.compile(r"(r\d)")
    files = sorted(Path(flood_data_dir).glob('**/df_flood_filtered.feather'))
    scriptLogger.info(f"Found {len(files)} files to regroup")
    projected_files = [f for f in files if 'SSP3' not in f.parent.name]
    historic_files = [f for f in files if 'SSP3' in f.parent.name]
    models = [f.parent.name for f in files if 'SSP3' not in f.parent.name]
    models = [sc_re.search(sc).group(1) for sc in models]
    scriptLogger.info(f"Found the following models : {models}")
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
    (output/"builded-data"/mrio_fullname).mkdir(parents=True, exist_ok=True)
    scriptLogger.info("Saving pre-clustered floods")
    save_parquet_with_meta(df, output/"builded-data"/mrio_fullname/f"1_pre-clustered_floods_{name}_{mrio_fullname}_with_dmg.parquet")
    flopros = flopros.to_crs(crs=3857)
    scriptLogger.info("Filtering floods with protection layer")
    df = add_prot(df,flopros)
    save_parquet_with_meta(df,output/"builded-data"/mrio_fullname/f"2_pre-clustered_floods_{name}_{mrio_fullname}_with_dmg_with_prot.parquet")
    check_na(df[['final_cluster', 'model', 'mrio', 'mrio_region', 'date', 'dmg', 'aff_pop', 'long', 'lat', 'return_period', "MerL_Riv"]])
    df = df[~df["protected"]]
    tmp_df = df[['model','final_cluster','mrio','mrio_region','long','lat','return_period','date','dmg','aff_pop']].copy()
    scriptLogger.info("Regrouping flood in same cell and close date (by MRIOT regions)")
    df = regroup_same_cell_diff_dates(tmp_df)
    df.reset_index(inplace=True)
    tmp_df = df[['model', 'final_cluster', 'mrio', 'mrio_region', 'long', 'lat', 'return_period', 'date_start', 'date_end', 'time_extent', 'total_cell_dmg', 'aff_pop']].copy()
    scriptLogger.info("Regrouping flood by group of cells and same date (by MRIOT regions)")
    df = regroup_diff_cell_same_date_cluster(tmp_df)
    df.reset_index(inplace=True)
    df.time_extent = df.time_extent.round(0).astype(int)
    scriptLogger.info("Attributing damages to sectors")
    df[["residential_dmg","industrial_dmg","commercial_dmg","infrastructure_dmg"]] = compute_sector_shares(df,shares)
    df["total_capital_dmg"] = df['total_event_dmg'] * K_DMG_SHARE
    df.rename(columns={"total_capital_dmg":"Total direct damage to capital (2010€PPP)","total_pop":"Population aff (2015 est.)"},inplace=True)
    save_parquet_with_meta(df,output/"builded-data"/mrio_fullname/f"3_clustered_floods_{name}_{mrio_fullname}.parquet")

def period_change(df_hist, df_proj,output,mrio_name):
    df_hist = pd.concat([df_hist,df_proj.loc[(df_proj.date_start.dt.year<JUNCTION_YEAR)]],axis=0)
    df_proj = df_proj.loc[(df_proj.date_start.dt.year>=JUNCTION_YEAR)].copy()
    df_hist["final_cluster"] = df_hist["final_cluster"] + "_" + df_hist["date_start"].dt.year.astype('str')
    df_proj["final_cluster"] = df_proj["final_cluster"] + "_" + df_proj["date_start"].dt.year.astype('str')
    save_parquet_with_meta(df_hist,output/"builded-data"/mrio_name/f"4_clustered_floods_{HIST_PERIOD_NAME}_{mrio_name}.parquet")
    save_parquet_with_meta(df_proj,output/"builded-data"/mrio_name/f"4_clustered_floods_{PROJ_PERIOD_NAME}_{mrio_name}.parquet")

def direct_prodloss(df,va_df, event_template, ref_year):
    scriptLogger.info(event_template)
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

    #df['time_extent'] = (df['date_end'] - df['date_start']).dt.days + 1
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

def compute_dmg_as_gva_share(df, mrio_ref):
    gva_df = (mrio_ref.x.T - mrio_ref.Z.sum(axis=0)).groupby("region",axis=1).sum().sum(axis=0)
    try:
        factor = mrio_ref.monetary_factor
    except AttributeError:
        scriptLogger.warning("No monetary factor associated with reference MRIOT, assuming 10^6.")
        factor = 10**6

    df['house_dmg_as_refyear_gva_share'] = df.groupby("mrio_region",axis=0)["residential_dmg"].transform(lambda x : x  / (gva_df[x.name] * factor))
    df['prod_capital_dmg_as_refyear_gva_share'] = df.groupby("mrio_region",axis=0)["Total direct damage to capital (2010€PPP)"].transform(lambda x : x  / (gva_df[x.name] * factor))
    df['tot_dmg_as_refyear_gva_share'] = df.groupby("mrio_region",axis=0)["total_event_dmg"].transform(lambda x : x  / (gva_df[x.name] * factor))
    return df

def global_treatment_after_period_change(df, mrio_name, mrio_ref, output_dir, period_name):
    mrio_re = re.compile(r"^(?P<mrio_basename>[a-zA-Z0-9]+)(?:_(?P<mrio_subname>full|\d+_sectors))?$")
    match = mrio_re.match(mrio_name)
    if not match:
        raise ValueError(f"{mrio_name} is not a valid mrio")

    # mrio_basename = match['mrio_basename']
    # mrio_subname = "full" if match['mrio_subname'] is None else match["mrio_subname"]

    df = compute_dmg_as_gva_share(df, mrio_ref)


    #scriptLogger.info("Computing direct production losses")
    #df = compute_direct_prodloss(df,gva_df,va_df,event_template,ref_year)
    save_parquet_with_meta(df,output_dir/"builded-data"/mrio_name/f"5_clustered_floods_{period_name}_{mrio_name}_with_prodloss.parquet")
    return output_dir/"builded-data"/mrio_name/f"5_clustered_floods_{period_name}_{mrio_name}_with_prodloss.parquet"

def load_mrio(filename: str, pkl_filepath) -> pym.IOSystem:
    """
    Loads the pickle file with the given filename.

    Args:
        filename: A string representing the name of the file to load (without the .pkl extension).
                  Valid file names follow the format <prefix>_full_<year>, where <prefix> is one of
                  'oecd_v2021', 'euregio', 'exiobase3', or 'eora26', and <year> is a four-digit year
                  such as '2000' or '2010'.

    Returns:
        The loaded pickle file.

    Raises:
        ValueError: If the given filename does not match the valid file name format, or the file doesn't contain an IOSystem.

    """
    regex = re.compile(
        r"^(oecd_v2021|euregio|exiobase3|eora26)_full_(\d{4})"
    )  # the regular expression to match filenames

    match = regex.match(filename)  # match the filename with the regular expression

    if not match:
        raise ValueError(f"The file name {filename} is not valid.")

    prefix, year = match.groups()  # get the prefix and year from the matched groups

    pkl_filepath = Path(pkl_filepath)

    fullpath = pkl_filepath / prefix /  f"{filename}.pkl"  # create the full file path

    scriptLogger.info(f"Loading {filename} mrio")
    with open(fullpath, "rb") as f:
        mrio = pickle.load(f)  # load the pickle file

    if not isinstance(mrio, pym.IOSystem):
        raise ValueError(f"{filename} was loaded but it is not an IOSystem")

    return mrio


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

def select_period(df,start,end) -> pd.DataFrame:
    return df.loc[(df.date_start.dt.year>=start) & (df.date_start.dt.year<=end)]

def filter_gva_max(df,max_gva, based_on_column) -> pd.DataFrame:
    return df.loc[df[based_on_column]<=max_gva]

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
def find_samples(df, pct_cls, column):
    df_copy = df.copy()
    df_copy = df_copy.reset_index()
    res = None
    percentiles = df_copy.groupby(['mrio_region'])[[column]].describe(percentiles=[0.01,0.05,0.1,0.33,0.5,0.66,0.75,0.8,0.9,0.99])
    i=0
    for region in df_copy.mrio_region.unique():
        i+=1
        for cl in pct_cls:
            val = percentiles.loc[region,pd.IndexSlice[:,cl]].values[0]
            tmp = (df_copy.iloc[find_neighbours(val, df_copy.loc[(df_copy["mrio_region"]==region)], column)]).copy()
            tmp.loc[:,"class"] = cl
            if res is None:
                res = tmp.copy()
            else:
                res = pd.concat([res,tmp])
        print("Region {} done! {}/{}".format(region,i,len(df_copy.mrio_region.unique())))
    return res

def build_rep_events_from_df(df, based_on_column):
    ddf = find_samples(df, ['min', "1%", "5%", "10%", "33%", "50%", "66%", "75%", "80%", "90%", "99%", "max"], based_on_column)
    ddf = ddf.drop_duplicates(subset=["mrio_region",based_on_column]).sort_values(by=["mrio_region",'class'])
    return ddf

def build_rep_events_from_parquet(parquet, based_on_column, gva_max=None):
    parquet = Path(parquet)
    df = pd.read_parquet(parquet)
    ddf = build_rep_events_from_df(df, based_on_column)
    if gva_max is not None:
        ddf = filter_gva_max(ddf, gva_max, based_on_column)
    return ddf
