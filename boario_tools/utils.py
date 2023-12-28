import pandas as pd
import geopandas as gpd
from datetime import datetime
import pathlib
import json
import os
import argparse
import pyarrow as pa

def check_na(df: pd.DataFrame):
    """Asserts a DataFrame has no NA

    Parameters
    ----------
    df : DataFrame
        The DataFrame to check

    """
    assert df.isna().sum().sum() == 0


def gdfy_from_long_lat(df: pd.DataFrame, longitude_col='long', latitude_col='lat', crs="epsg:4326"):
    """Creates a GeoDataFrame from a DataFrame with longitude and latitude columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with coordinates.
    longitude_col : str
        The column name containing longitude values.
    latitude_col : str
        The column name containing latitude values.
    crs : str
        The coordinate reference system description.

    Returns
    -------
    gpd.GeoDataFrame
        The GeoDataFrame with geometry based on the provided coordinates.
    """
    geometry = gpd.points_from_xy(df[longitude_col], df[latitude_col])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    return gdf

def save_parquet_with_meta(df, path, meta_suffix="_meta", meta_extension=".json"):
    """Saves a DataFrame to a parquet file with associated metadata.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be saved.
    path : str or pathlib.Path
        The path to save the parquet file.
    meta_suffix : str, optional
        The suffix for the metadata file (default is "_meta").
    meta_extension : str, optional
        The file extension for the metadata file (default is ".json").
    """
    path = pathlib.Path(path).resolve()

    # Prepare metadata
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    meta = {
        "last_updated": now,
        "file": path.name,
        "pandas_version": pd.__version__,
        "pyarrow_version": pa.__version__,
    }

    # Save metadata
    meta_path = path.with_stem(path.stem + meta_suffix).with_suffix(meta_extension)
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=4)

    try:
        # Save DataFrame to parquet
        df.to_parquet(path)
        print(f"DataFrame saved to {path}")
    except Exception as e:
        print(f"Error saving DataFrame to {path}: {e}")

def read_parquet_with_meta(path, meta_suffix="_meta", meta_extension=".json"):
    """Reads a DataFrame from a parquet file with associated metadata.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to the parquet file.
    meta_suffix : str, optional
        The suffix for the metadata file (default is "_meta").
    meta_extension : str, optional
        The file extension for the metadata file (default is ".json").

    Returns
    -------
    pd.DataFrame
        The DataFrame read from the parquet file.
    """
    path = pathlib.Path(path).resolve()

    # Determine metadata path
    meta_path = path.with_stem(path.stem + meta_suffix).with_suffix(meta_extension)

    try:
        # Read metadata
        with meta_path.open("r") as f:
            meta = json.load(f)
        print(meta)

        # Read DataFrame from parquet
        df = pd.read_parquet(path)
        print(f"DataFrame loaded from {path}")

        return df
    except FileNotFoundError:
        print(f"Metadata file not found: {meta_path}")
        return None
    except Exception as e:
        print(f"Error reading DataFrame from {path}: {e}")
        return None


def dir_path(path):
    """Check if the given path is a readable directory.

    Parameters
    ----------
    path : str
        The path to be checked.

    Returns
    -------
    str
        The valid directory path.

    Raises
    ------
    argparse.ArgumentTypeError
        If the path is not a valid or readable directory.
    """
    path = os.path.abspath(os.path.expanduser(path))

    if os.path.isdir(path) and os.access(path, os.R_OK):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid or readable directory.")


def file_path(path):
    """Check if the given path points to an existing file.

    Parameters
    ----------
    path : str
        The path to be checked.

    Returns
    -------
    str
        The valid file path.

    Raises
    ------
    argparse.ArgumentTypeError
        If the file does not exist.
    """
    path = os.path.abspath(os.path.expanduser(path))

    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"The file '{path}' does not exist.")
