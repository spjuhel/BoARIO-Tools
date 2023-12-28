import pytest

import os
import argparse
import geopandas as gpd
import pandas as pd
import pathlib

from boario_tools.utils import check_na, gdfy_from_long_lat, save_parquet_with_meta, read_parquet_with_meta, dir_path, file_path

def test_check_na():
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    check_na(df)  # This should not raise an exception

    df_with_na = pd.DataFrame({'col1': [1, 2, None], 'col2': [4, 5, 6]})
    with pytest.raises(AssertionError):
        check_na(df_with_na)

def test_gdfy_from_long_lat():
    df = pd.DataFrame({'long': [1.0, 2.0, 3.0], 'lat': [4.0, 5.0, 6.0]})
    result = gdfy_from_long_lat(df)
    assert isinstance(result, gpd.GeoDataFrame)

def test_save_and_read_parquet_with_meta(tmp_path):
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    file_path = tmp_path / "test_data.parquet"

    save_parquet_with_meta(df, file_path)

    loaded_df = read_parquet_with_meta(file_path)
    pd.testing.assert_frame_equal(df, loaded_df)

def test_dir_path():
    valid_dir = os.getcwd()  # Use the current working directory as a valid directory
    assert dir_path(valid_dir) == valid_dir

    invalid_dir = "invalid_directory"
    with pytest.raises(argparse.ArgumentTypeError):
        dir_path(invalid_dir)  # This should raise an exception

def test_file_path(tmp_path):
    valid_file : pathlib.PosixPath = tmp_path / "test_file.txt"
    valid_file.touch()

    assert file_path(valid_file) == str(valid_file)

    invalid_file = tmp_path / "nonexistent_file.txt"
    with pytest.raises(argparse.ArgumentTypeError):
        file_path(invalid_file)  # This should raise an exception
