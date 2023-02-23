import pandas as pd
import geopandas as gpd
from datetime import datetime
import pathlib
import json
from pyarrow import __version__ as pyarrowver

def check_na(df):
    assert df.isna().sum().sum() == 0

def gdfy_from_long_lat(df:pd.DataFrame, crs="epsg:4326"):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat), crs = crs)
    return gdf

def save_parquet_with_meta(df, path):
    path = pathlib.Path(path).resolve()
    print("Saving df to {}".format(path))
    meta_path = path.with_stem(path.stem+"_meta").with_suffix(".json")
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    meta = {
        "last_updated":now,
        "file": path.name,
        "pandas_version":pd.__version__,
        "pyarrow_version":pyarrowver,
    }
    with pathlib.Path(meta_path).open('w') as f:
        json.dump(meta,f, indent=4)
    df.to_parquet(path)

def read_parquet_with_meta(path):
    path = pathlib.Path(path).resolve()
    meta_path = path.with_stem(path.stem+"_meta").with_suffix(".json")
    with pathlib.Path(meta_path).open('r') as f:
        meta = json.load(f)
    print(meta)
    return pd.read_parquet(path)
