import math
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pymrio
import pickle as pkl

import requests
from boario_tools import LOGGER
import copy

import warnings
from importlib import resources
from boario_tools.regex_patterns import MRIOT_FULLNAME_REGEX, MRIOT_YEAR_REGEX

from collections.abc import Iterable
from typing import cast
import pathlib
import zipfile

from tqdm import tqdm
from .config import config

POSSIBLE_MRIOT_REGEXP = MRIOT_FULLNAME_REGEX
EUREGIO_REGIONS_RENAMING = {"DEE1": "DEE0", "DEE2": "DEE0", "DEE3": "DEE0"}

MRIOT_DIRECTORY = Path(config.get("settings", "data_folder", "/tmp/boario-data/"))
"""Directory where Multi-Regional Input-Output Tables (MRIOT) are downloaded."""

MRIOT_TYPE_REGEX = (
    r"(?P<mrio_type>OECD23|EXIOBASE3|EORA26|WIOD16)"
 )

MRIOT_DEFAULT_FILENAME = {
    "EXIOBASE3": lambda year: f"IOT_{year}_ixi.zip",
    "WIOD16": lambda year: f"WIOT{year}_Nov16_ROW.xlsb",
    "OECD23": lambda year: f"ICIO2023_{year}.csv",
    "EORA26": lambda year: f"Eora26_{year}_bp.zip"
}

MRIOT_MONETARY_FACTOR = {
    "EXIOBASE3": 1000000,
    "EORA26": 1000,
    "WIOD16": 1000000,
    "OECD23": 1000000,
    "EUREGIO": 1000000,
}

MRIOT_COUNTRY_CONVERTER_CORR = {
    "EXIOBASE3" : "EXIO3",
    "WIOD16" : "WIOD",
    "EORA26" : "Eora",
    "OECD23" : "ISO3"
}

WIOD_FILE_LINK = config.get("resources", "wiod16_url")
"""Link to the 2016 release of the WIOD tables."""

VA_NAME = "value added"
"""Index name for value added"""

_ATTR_LIST = [
    "Z",
    "Y",
    "x",
    "A",
    "As",
    "G",
    "L",
    "unit",
    "population",
    "meta",
    "__non_agg_attributes__",
    "__coefficients__",
    "__basic__",
]

ICIO23_sectors_mapping = {'A01_02': 'Agriculture, hunting, forestry',
 'A03': 'Fishing and aquaculture',
 'B05_06': 'Mining and quarrying, energy producing products',
 'B07_08': 'Mining and quarrying, non-energy producing products',
 'B09': 'Mining support service activities',
 'C10T12': 'Food products, beverages and tobacco',
 'C13T15': 'Textiles, textile products, leather and footwear',
 'C16': 'Wood and products of wood and cork',
 'C17_18': 'Paper products and printing',
 'C19': 'Coke and refined petroleum products',
 'C20': 'Chemical and chemical products',
 'C21': 'Pharmaceuticals, medicinal chemical and botanical products',
 'C22': 'Rubber and plastics products',
 'C23': 'Other non-metallic mineral products',
 'C24': 'Basic metals',
 'C25': 'Fabricated metal products',
 'C26': 'Computer, electronic and optical equipment',
 'C27': 'Electrical equipment',
 'C28': 'Machinery and equipment, nec ',
 'C29': 'Motor vehicles, trailers and semi-trailers',
 'C30': 'Other transport equipment',
 'C31T33': 'Manufacturing nec; repair and installation of machinery and equipment',
 'D': 'Electricity, gas, steam and air conditioning supply',
 'E': 'Water supply; sewerage, waste management and remediation activities',
 'F': 'Construction',
 'G': 'Wholesale and retail trade; repair of motor vehicles',
 'H49': 'Land transport and transport via pipelines',
 'H50': 'Water transport',
 'H51': 'Air transport',
 'H52': 'Warehousing and support activities for transportation',
 'H53': 'Postal and courier activities',
 'I': 'Accommodation and food service activities',
 'J58T60': 'Publishing, audiovisual and broadcasting activities',
 'J61': 'Telecommunications',
 'J62_63': 'IT and other information services',
 'K': 'Financial and insurance activities',
 'L': 'Real estate activities',
 'M': 'Professional, scientific and technical activities',
 'N': 'Administrative and support services',
 'O': 'Public administration and defence; compulsory social security',
 'P': 'Education',
 'Q': 'Human health and social work activities',
 'R': 'Arts, entertainment and recreation',
 'S': 'Other service activities',
 'T': 'Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use'}


def conda_check():
    LOGGER.info(
        """Make sure you use the same python environment as the one loading
        the pickle file (especial pymrio and pandas version !)"""
    )
    try:
        LOGGER.info("Your current environment is: {}".format(os.environ["CONDA_PREFIX"]))
    except KeyError:
        LOGGER.info(
            "Could not find CONDA_PREFIX, this is normal if you are not using conda."
        )

############################# Main function ##############################

def get_mriot(mriot_type, mriot_year, redownload=False, save=True):
    # if data were parsed and saved: load them
    downloads_dir = MRIOT_DIRECTORY / mriot_type / "downloads"
    downloaded_file = downloads_dir / MRIOT_DEFAULT_FILENAME[mriot_type](mriot_year)
    # parsed data directory
    parsed_data_dir = MRIOT_DIRECTORY / mriot_type / str(mriot_year)

    if redownload and downloaded_file.exists():
        for fil in downloads_dir.iterdir():
            fil.unlink()
        downloads_dir.rmdir()
    if redownload and parsed_data_dir.exists():
        for fil in parsed_data_dir.iterdir():
            fil.unlink()
        parsed_data_dir.rmdir()

    if not downloaded_file.exists():
        download_mriot(mriot_type, mriot_year, downloads_dir)
    if not parsed_data_dir.exists():
        mriot = parse_mriot(mriot_type, downloaded_file, mriot_year)
        if save:
            mriot.save(parsed_data_dir, table_format="parquet")
    else:
        mriot = pymrio.load(path=parsed_data_dir)
        # Not too dirty trick to keep pymrio's saver/loader but have additional attributes.
        setattr(mriot, "monetary_factor", mriot.meta._content["monetary_factor"])
        setattr(mriot, "basename", mriot.meta._content["basename"])
        setattr(mriot, "year", mriot.meta._content["year"])
        setattr(mriot, "sectors_agg", mriot.meta._content["sectors_agg"])
        setattr(mriot, "regions_agg", mriot.meta._content["regions_agg"])

    return mriot


####################### Downloading and parsing ######################################

def download_file(url, download_dir=None, overwrite=True):
    """Download file from url to given target folder and provide full path of the downloaded file.

    Parameters
    ----------
    url : str
        url containing data to download
    download_dir : Path or str, optional
        the parent directory of the eventually downloaded file
        default: local_data.save_dir as defined in climada.conf
    overwrite : bool, optional
        whether or not an already existing file at the target location should be overwritten,
        by default True

    Returns
    -------
    str
        the full path to the eventually downloaded file
    """
    file_name = url.split('/')[-1]
    if file_name.strip() == '':
        raise ValueError(f"cannot download {url} as a file")
    download_path = Path("/tmp/boario-downloads/") if download_dir is None else Path(download_dir)
    file_path = download_path.absolute().joinpath(file_name)
    if file_path.exists():
        if not file_path.is_file() or not overwrite:
            raise FileExistsError(f"cannot download to {file_path}")

    try:
        req_file = requests.get(url, stream=True)
    except IOError as ioe:
        raise type(ioe)('Check URL and internet connection: ' + str(ioe)) from ioe
    if req_file.status_code < 200 or req_file.status_code > 299:
        raise ValueError(f'Error loading page {url}\n'
                         f' Status: {req_file.status_code}\n'
                         f' Content: {req_file.content}')

    total_size = int(req_file.headers.get('content-length', 0))
    block_size = 1024

    LOGGER.info('Downloading %s to file %s', url, file_path)
    with file_path.open('wb') as file:
        for data in tqdm(req_file.iter_content(block_size),
                         total=math.ceil(total_size // block_size),
                         unit='KB', unit_scale=True):
            file.write(data)

    return str(file_path)


def download_mriot(mriot_type, mriot_year, download_dir):
    """Download EXIOBASE3, WIOD16 or OECD23 Multi-Regional Input Output Tables
    for specific years.

    Parameters
    ----------
    mriot_type : str
    mriot_year : int
    download_dir : pathlib.PosixPath

    Notes
    -----
    The download of EXIOBASE3 and OECD23 tables makes use of pymrio functions.
    The download of WIOD16 tables requires ad-hoc functions, since the
    related pymrio functions were broken at the time of implementation
    of this function.
    """

    if mriot_type == "EXIOBASE3":
        pymrio.download_exiobase3(
            storage_folder=download_dir, system="ixi", years=[mriot_year]
        )

    elif mriot_type == "WIOD16":
        download_dir.mkdir(parents=True, exist_ok=True)
        downloaded_file_name = download_file(
            WIOD_FILE_LINK,
            download_dir=download_dir,
        )
        downloaded_file_zip_path = pathlib.Path(downloaded_file_name + ".zip")
        pathlib.Path(downloaded_file_name).rename(downloaded_file_zip_path)

        with zipfile.ZipFile(downloaded_file_zip_path, "r") as zip_ref:
            zip_ref.extractall(download_dir)

    elif mriot_type == "OECD23":
        years_groups = ["1995-2000", "2001-2005", "2006-2010", "2011-2015", "2016-2020"]
        year_group = years_groups[int(np.floor((mriot_year - 1995) / 5))-1]
        pymrio.download_oecd(storage_folder=download_dir, years=year_group)

    else:
        raise ValueError(f"Invalid MRIOT type {mriot_type}")


def parse_mriot(mriot_type, downloaded_file, mriot_year, **kwargs):
    """Parse EXIOBASE3, WIOD16 or OECD23 MRIOT for specific years

    Parameters
    ----------
    mriot_type : str
    downloaded_file : pathlib.PosixPath

    Notes
    -----
    The parsing of EXIOBASE3 and OECD23 tables makes use of pymrio functions.
    The parsing of WIOD16 tables requires ad-hoc functions, since the
    related pymrio functions were broken at the time of implementation
    of this function.

    Some metadata is rewrote or added to the objects for consistency in usage (name, monetary factor, year).
    """

    if mriot_type == "EXIOBASE3":
        mriot = build_exio3_from_zip(mrio_zip=downloaded_file, **kwargs)
    elif mriot_type == "WIOD16":
        mriot = parse_wiod_v2016(mrio_xlsb=downloaded_file)
    elif mriot_type == "OECD23":
        mriot = build_oecd_from_csv(mrio_csv=downloaded_file, year=mriot_year)
    elif mriot_type == "EORA26":
        mriot = build_eora_from_zip(mrio_zip=downloaded_file, **kwargs)
    else:
        raise RuntimeError(f"Unknown mriot_type: {mriot_type}")

    mriot.meta.change_meta(
        "description", "Metadata for pymrio Multi Regional Input-Output Table"
    )
    mriot.meta.change_meta("name", f"{mriot_type}-{mriot_year}")

    # Check if negative demand - this happens when the
    # "Changes in Inventory (CII)" demand category is
    # larger than the sum of all other categories
    if (mriot.Y.sum(axis=1) < 0).any():
        warnings.warn(
            "Found negatives values in total final demand, "
            "setting them to 0 and recomputing production vector"
        )
        mriot.Y.loc[mriot.Y.sum(axis=1) < 0] = mriot.Y.loc[
            mriot.Y.sum(axis=1) < 0
        ].clip(lower=0)
        mriot.x = pymrio.calc_x(mriot.Z, mriot.Y)
        mriot.A = pymrio.calc_A(mriot.Z, mriot.x)

    return mriot

def build_exio3_from_zip(mrio_zip: str, remove_attributes=True, aggregate_ROW=True):
    mrio_path = pathlib.Path(mrio_zip)
    mrio_pym = pymrio.parse_exiobase3(path=mrio_path)
    mrio_pym = cast(pymrio.IOSystem, mrio_pym)  # Just for the LSP
    if remove_attributes:
        LOGGER.info("Removing unnecessary IOSystem attributes")
        attr = _ATTR_LIST
        tmp = list(mrio_pym.__dict__.keys())
        for at in tmp:
            if at not in attr:
                delattr(mrio_pym, at)
        LOGGER.info("Done")

    mrio_pym.meta.change_meta("name", "EXIOBASE3")

    if aggregate_ROW:
        LOGGER.info("Aggregating the different ROWs regions together")
        agg_regions = pd.DataFrame(
            {
                "original": mrio_pym.get_regions()[
                    ~mrio_pym.get_regions().isin(["WA", "WE", "WF", "WL", "WM"])
                ].tolist()
                + ["WA", "WE", "WF", "WL", "WM"],
                "aggregated": mrio_pym.get_regions()[
                    ~mrio_pym.get_regions().isin(["WA", "WE", "WF", "WL", "WM"])
                ].tolist()
                + ["ROW"] * 5,
            }
        )
        mrio_pym = mrio_pym.aggregate(region_agg=agg_regions)

    LOGGER.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    LOGGER.info("Done")

    LOGGER.info("Computing the missing IO components")
    mrio_pym.calc_all()
    LOGGER.info("Done")

    setattr(mrio_pym, "monetary_factor", MRIOT_MONETARY_FACTOR["EXIOBASE3"])
    setattr(mrio_pym, "basename", "exiobase3_ixi")
    setattr(mrio_pym, "year", mrio_pym.meta.description[-4:])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    # Also put it in meta for saving
    mrio_pym.meta.change_meta("monetary_factor", MRIOT_MONETARY_FACTOR["EXIOBASE3"])
    mrio_pym.meta.change_meta("year", mrio_pym.meta.description[-4:])
    mrio_pym.meta.change_meta("basename", "exiobase3_ixi")
    mrio_pym.meta.change_meta("sectors_agg", "full_sectors")
    mrio_pym.meta.change_meta("regions_agg", "full_regions")
    return mrio_pym


def build_eora_from_zip(
    mrio_zip: str,
    reexport_treatment=False,
    inv_treatment=True,
    remove_attributes=True,
):
    mrio_path = pathlib.Path(mrio_zip)
    mrio_pym = pymrio.parse_eora26(path=mrio_path)
    LOGGER.info("Removing unnecessary IOSystem attributes")
    if remove_attributes:
        attr = _ATTR_LIST
        tmp = list(mrio_pym.__dict__.keys())
        for at in tmp:
            if at not in attr:
                delattr(mrio_pym, at)
    LOGGER.info("Done")

    setattr(mrio_pym, "monetary_factor", MRIOT_MONETARY_FACTOR["EORA26"])
    setattr(mrio_pym, "basename", "eora26")
    setattr(mrio_pym, "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    # Also put it in meta for saving
    mrio_pym.meta.change_meta("monetary_factor", MRIOT_MONETARY_FACTOR["EORA26"])
    mrio_pym.meta.change_meta(
        "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"]
    )
    mrio_pym.meta.change_meta("basename", "eora26")
    mrio_pym.meta.change_meta("sectors_agg", "full_sectors")
    mrio_pym.meta.change_meta("regions_agg", "full_regions")

    if reexport_treatment:
        LOGGER.info(
            "EORA26 has the re-import/re-export sector which other mrio often don't have (ie EXIOBASE), we put it in 'Other'."
        )
        mrio_pym.rename_sectors({"Re-export & Re-import": "Others"})
        mrio_pym.aggregate_duplicates()
        setattr(mrio_pym, "sectors_agg", "full_no_reexport_sectors")

    if inv_treatment:
        LOGGER.info(
            "EORA26 has negative values in its final demand which can cause problems. We set them to 0."
        )
        if mrio_pym.Y is not None:
            mrio_pym.Y = mrio_pym.Y.clip(lower=0)
        else:
            raise AttributeError("Y attribute is not set")

    LOGGER.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    LOGGER.info("Done")

    LOGGER.info("Computing the missing IO components")
    mrio_pym.calc_all()
    LOGGER.info("Done")

    return mrio_pym


def build_oecd_from_csv(
    mrio_csv: str, year: int | None = None, remove_attributes: bool = True
):
    mrio_path = pathlib.Path(mrio_csv)
    mrio_pym = pymrio.parse_oecd(path=mrio_path, year=year)
    # Drop the "ALL" column for consistency
    mrio_pym.Y = mrio_pym.Y.drop("ALL",axis=1)
    LOGGER.info("Removing unnecessary IOSystem attributes")
    if remove_attributes:
        attr = _ATTR_LIST
        tmp = list(mrio_pym.__dict__.keys())
        for at in tmp:
            if at not in attr:
                delattr(mrio_pym, at)
    LOGGER.info("Done")
    setattr(mrio_pym, "monetary_factor", MRIOT_MONETARY_FACTOR["OECD23"])
    setattr(mrio_pym, "basename", "icio_v2023")
    setattr(mrio_pym, "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    # Also put it in meta for saving
    mrio_pym.meta.change_meta("monetary_factor", MRIOT_MONETARY_FACTOR["OECD23"])
    mrio_pym.meta.change_meta(
        "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"]
    )
    mrio_pym.meta.change_meta("basename", "icio_v2023")
    mrio_pym.meta.change_meta("sectors_agg", "full_sectors")
    mrio_pym.meta.change_meta("regions_agg", "full_regions")

    LOGGER.info("Computing the missing IO components")
    mrio_pym.calc_all()
    LOGGER.info("Done")
    LOGGER.info("Renaming sectors")
    mrio_pym.rename_sectors(ICIO23_sectors_mapping)
    LOGGER.info("Done")
    LOGGER.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    LOGGER.info("Done")
    return mrio_pym


def parse_wiod_v2016(mrio_xlsb: str):
    mrio_path = pathlib.Path(mrio_xlsb)
    mriot_df = pd.read_excel(mrio_xlsb, engine="pyxlsb")
    Z, Y, x = parse_mriot_from_df(
        mriot_df,
        col_iso3=2,
        col_sectors=1,
        row_fd_cats=2,
        rows_data=(5, 2469),
        cols_data=(4, 2468),
    )
    mrio_pym = pymrio.IOSystem(Z=Z, Y=Y, x=x)
    multiindex_unit = pd.MultiIndex.from_product(
        [mrio_pym.get_regions(), mrio_pym.get_sectors()], names=["region", "sector"]  # type: ignore
    )
    mrio_pym.unit = pd.DataFrame(
        data=np.repeat(["M.USD"], len(multiindex_unit)),
        index=multiindex_unit,
        columns=["unit"],
    )

    setattr(mrio_pym, "monetary_factor", MRIOT_MONETARY_FACTOR["WIOD16"])
    setattr(mrio_pym, "basename", "wiod_v2016")
    setattr(mrio_pym, "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    # Also put it in meta for saving
    mrio_pym.meta.change_meta("monetary_factor", MRIOT_MONETARY_FACTOR["WIOD16"])
    mrio_pym.meta.change_meta(
        "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"]
    )
    mrio_pym.meta.change_meta("basename", "wiod_v2016")
    mrio_pym.meta.change_meta("sectors_agg", "full_sectors")
    mrio_pym.meta.change_meta("regions_agg", "full_regions")

    LOGGER.info("Computing the missing IO components")
    mrio_pym.calc_all()
    LOGGER.info("Done")
    LOGGER.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    LOGGER.info("Done")
    return mrio_pym


def parse_mriot_from_df(
    mriot_df, col_iso3, col_sectors, rows_data, cols_data, row_fd_cats=None
):
    """Build multi-index dataframes of the transaction matrix, final demand and total
       production from a Multi-Regional Input-Output Table dataframe.

    Parameters
    ----------
    mriot_df : pandas.DataFrame
        The Multi-Regional Input-Output Table
    col_iso3 : int
        Column's position of regions' iso names
    col_sectors : int
        Column's position of sectors' names
    rows_data : (int, int)
        Tuple of integers with positions of rows
        containing the MRIOT data for intermediate demand
        matrix.
        Final demand matrix is assumed to be the remaining columns
        of the DataFrame except the last one (which generally holds
        total output).
    cols_data : (int, int)
        Tuple of integers with positions of columns
        containing the MRIOT data
    row_fd_cats : int
        Integer index of the row containing the
        final demand categories.
    """

    start_row, end_row = rows_data
    start_col, end_col = cols_data

    sectors = mriot_df.iloc[start_row:end_row, col_sectors].unique()
    regions = mriot_df.iloc[start_row:end_row, col_iso3].unique()
    if row_fd_cats is None:
        n_fd_cat = (mriot_df.shape[1] - (end_col + 1)) // len(regions)
        fd_cats = [f"fd_cat_{i}" for i in range(n_fd_cat)]
    else:
        fd_cats = mriot_df.iloc[row_fd_cats, end_col:-1].unique()

    multiindex = pd.MultiIndex.from_product(
        [regions, sectors], names=["region", "sector"]
    )

    multiindex_final_demand = pd.MultiIndex.from_product(
        [regions, fd_cats], names=["region", "category"]
    )

    Z = mriot_df.iloc[start_row:end_row, start_col:end_col].values.astype(float)
    Z = pd.DataFrame(data=Z, index=multiindex, columns=multiindex)

    Y = mriot_df.iloc[start_row:end_row, end_col:-1].values.astype(float)
    Y = pd.DataFrame(data=Y, index=multiindex, columns=multiindex_final_demand)

    x = mriot_df.iloc[start_row:end_row, -1].values.astype(float)
    x = pd.DataFrame(data=x, index=multiindex, columns=["indout"])

    return Z, Y, x


def euregio_convert_xlsx2csv(inpt, out_folder, office_exists):
    if not office_exists:
        raise FileNotFoundError(
            "Creating csvs files require libreoffice which wasn't found. You may wan't to convert EUREGIO files by yourself if you are unable to install libreoffice"
        )
    LOGGER.info(
        f"Executing: libreoffice --convert-to 'csv:Text - txt - csv (StarCalc):44,34,0,1,,,,,,,,3' {out_folder} {inpt}"
    )
    os.system(
        f"libreoffice --convert-to 'csv:Text - txt - csv (StarCalc):44,34,0,1,,,,,,,,3' --outdir {out_folder} {inpt}"
    )
    filename = Path(inpt).name
    LOGGER.info(filename)
    new_filename = (
        "euregio_" + filename.split("_")[1].split(".")[0].replace("-", "_") + ".csv"
    )
    old_path = Path(out_folder) / filename.replace(
        ".xlsb", "-{}.csv".format(filename.split("_")[1].split(".")[0])
    )
    new_path = Path(out_folder) / new_filename
    LOGGER.info(f"Executing: mv {old_path} {new_path}")
    os.rename(old_path, new_path)


def build_euregio_components_from_csv(csv_file, inv_treatment=True):
    cols_z = [2, 5] + [i for i in range(6, 3730, 1)]
    ioz = pd.read_csv(
        csv_file,
        index_col=[0, 1],
        usecols=cols_z,
        engine="c",
        names=None,
        header=None,
        skiprows=8,
        nrows=3724,
        decimal=".",
        low_memory=False,
    )
    ioz.rename_axis(index=["region", "sector"], inplace=True)
    ioz.columns = ioz.index
    ioz.fillna(value=0.0, inplace=True)

    cols_y = [3733, 3736] + [i for i in range(3737, 3737 + 1064, 1)]
    fd_index = pd.read_csv(
        csv_file, usecols=[3737, 3738, 3739, 3740], skiprows=7, header=0, nrows=0
    ).columns
    ioy = pd.read_csv(
        csv_file,
        index_col=[0, 1],
        usecols=cols_y,
        engine="c",
        names=None,
        header=None,
        skiprows=8,
        nrows=3724,
        decimal=".",
        low_memory=False,
    )
    ioy.rename_axis(index=["region", "sector"], inplace=True)
    ioy.columns = pd.MultiIndex.from_product(
        [ioy.index.get_level_values(0).unique(), fd_index]
    )
    ioy.fillna(value=0.0, inplace=True)

    iova = pd.read_csv(
        csv_file,
        index_col=[5],
        engine="c",
        header=[0, 3],
        skiprows=3735,
        nrows=6,
        decimal=".",
        low_memory=False,
    )
    iova.rename_axis(index=["va_cat"], inplace=True)
    iova.fillna(value=0.0, inplace=True)
    iova.drop(iova.iloc[:, :5].columns, axis=1, inplace=True)
    iova.drop(iova.iloc[:, 3724:].columns, axis=1, inplace=True)

    ioy = ioy.rename_axis(["region", "sector"])
    ioy = ioy.rename_axis(["region", "category"], axis=1)
    if inv_treatment:
        ioy = ioy.clip(lower=0)

    return ioz, ioy, iova


def build_euregio_from_csv(mrio_csv: str, year, correct_regions):
    conda_check()
    ioz, ioy, iova = build_euregio_components_from_csv(mrio_csv, inv_treatment=True)
    euregio = pymrio.IOSystem(
        Z=ioz,
        Y=ioy,
        year=year,
        unit=pd.DataFrame(
            data=["2010_€_MILLIONS"] * len(iova.index),
            index=iova.index,
            columns=["unit"],
        ),
    )
    setattr(euregio, "monetary_factor", 1000000)
    setattr(euregio, "basename", "euregio")
    setattr(euregio, "year", year)
    setattr(euregio, "sectors_agg", "full_sectors")
    setattr(euregio, "regions_agg", "full_regions")
    if correct_regions:
        LOGGER.info(f"Correcting germany regions : {EUREGIO_REGIONS_RENAMING}")
        euregio = euregio_correct_regions(euregio)
    LOGGER.info("Computing the missing IO components")
    euregio.calc_all()
    euregio.meta.change_meta("name", f"euregio {year}")

    assert isinstance(euregio, pymrio.IOSystem)
    LOGGER.info("Re-indexing lexicographicaly")
    euregio = lexico_reindex(euregio)
    LOGGER.info("Done")
    return euregio


def build_oecd_from_zip(mrio_zip: str, year: int):
    conda_check()
    mrio_path = Path(mrio_zip)
    mrio_pym = pymrio.parse_oecd(path=mrio_path, year=year)
    LOGGER.info("Removing unnecessary IOSystem attributes")
    attr = _ATTR_LIST
    tmp = list(mrio_pym.__dict__.keys())
    for at in tmp:
        if at not in attr:
            delattr(mrio_pym, at)
    assert isinstance(mrio_pym, pymrio.IOSystem)
    LOGGER.info("Done")
    setattr(mrio_pym, "monetary_factor", 1000000)
    setattr(mrio_pym, "basename", "icio_v2018")
    setattr(mrio_pym, "year", year)
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    LOGGER.info("Computing the missing IO components")
    mrio_pym.calc_all()
    LOGGER.info("Done")
    LOGGER.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    LOGGER.info("Done")

############################### IO, saving ###########################################
def euregio_csv_to_pkl(
    mrio_csv: str,
    output_dir: str,
    year,
    correct_regions=True,
    custom_name: str | None = None,
):
    mrio_pym = build_euregio_from_csv(mrio_csv, year, correct_regions)
    name = (
        custom_name
        if custom_name
        else f"{mrio_pym.basename}_{mrio_pym.year}_{mrio_pym.sectors_agg}_{mrio_pym.regions_agg}.pkl"
    )
    save_path = Path(output_dir) / name
    LOGGER.info("Saving to {}".format(save_path.absolute()))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pkl.dump(mrio_pym, f)


def eora26_zip_to_pkl(
    mrio_zip: str,
    output_dir: str,
    reexport_treatment=True,
    inv_treatment=True,
    remove_attributes=True,
    custom_name: str | None = None,
):
    mrio_pym = build_eora_from_zip(
        mrio_zip, reexport_treatment, inv_treatment, remove_attributes
    )
    name = (
        custom_name
        if custom_name
        else f"{mrio_pym.basename}_{mrio_pym.year}_{mrio_pym.sectors_agg}_{mrio_pym.regions_agg}.pkl"
    )
    save_path = Path(output_dir) / name
    LOGGER.info("Saving to {}".format(save_path.absolute()))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pkl.dump(mrio_pym, f)


def oecd_v2018_zip_to_pkl(
    mrio_zip: str, output_dir: str, year: int, custom_name: str | None = None
):
    mrio_pym = build_oecd_from_zip(mrio_zip, year)
    name = (
        custom_name
        if custom_name
        else f"{mrio_pym.basename}_{mrio_pym.year}_{mrio_pym.sectors_agg}_{mrio_pym.regions_agg}.pkl"
    )
    save_path = Path(output_dir) / name
    LOGGER.info("Saving to {}".format(save_path.absolute()))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pkl.dump(mrio_pym, f)


def wiod_v2016_xlsb2pkl(
    mrio_xlsb: str, output_dir: str, custom_name: str | None = None
):
    mrio_pym = parse_wiod_v2016(mrio_xlsb)
    name = (
        custom_name
        if custom_name
        else f"{mrio_pym.basename}_{mrio_pym.year}_{mrio_pym.sectors_agg}_{mrio_pym.regions_agg}.pkl"
    )
    save_path = Path(output_dir) / name
    LOGGER.info("Saving to {}".format(save_path.absolute()))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pkl.dump(mrio_pym, f)


def exio3_zip_to_pkl(
    mrio_zip: str,
    output_dir: str,
    remove_attributes: bool = True,
    custom_name: str | None = None,
):
    mrio_pym = build_exio3_from_zip(mrio_zip, remove_attributes)
    name = (
        custom_name
        if custom_name
        else f"{mrio_pym.basename}_{mrio_pym.year}_{mrio_pym.sectors_agg}_{mrio_pym.regions_agg}.pkl"
    )
    save_path = Path(output_dir) / name
    LOGGER.info("Saving to {}".format(save_path.absolute()))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pkl.dump(mrio_pym, f)


def load_mrio(
    filename: str, pkl_filepath, set_attribute_from_filename=False
) -> pymrio.IOSystem:
    """
    Loads the pickle file with the given filename.

    Args:
        filename: A string representing the name of the file to load (without the .pkl extension).
                  Valid file names follow the format <basename>_<year>_<suffix>, where <basename> is one of
                  'oecd_v2021', 'euregio', 'exiobase3_ixi', or 'eora26', and <year> is a four-digit year
                  such as '2000' or '2010'.

    Returns:
        The loaded pickle file.

    Raises:
        ValueError: If the given filename does not match the valid file name format, or the file doesn't contain an IOSystem.

    """
    regex = POSSIBLE_MRIOT_REGEXP
    rmatch = regex.match(filename)  # match the filename with the regular expression

    if not rmatch:
        raise ValueError(f"The file name {filename} is not valid.")

    (
        basename,
        year,
        sectors_agg,
        regions_agg,
    ) = (
        rmatch["mrio_basename"],
        rmatch["mrio_year"],
        rmatch["mrio_aggreg_sectors"],
        rmatch["mrio_aggreg_regions"],
    )  # get the basename and year from the matched groups

    pkl_filepath = Path(pkl_filepath)

    fullpath = pkl_filepath / basename / f"{filename}.pkl"  # create the full file path

    LOGGER.info(f"Loading {filename} mrio")
    with open(fullpath, "rb") as f:
        mriot = pkl.load(f)  # load the pickle file

    if not hasattr(mriot, "basename"):
        mriot.basename = None
    if not hasattr(mriot, "year"):
        mriot.year = None
    if not hasattr(mriot, "sectors_agg"):
        mriot.sectors_agg = None
    if not hasattr(mriot, "sectors_agg"):
        mriot.sectors_agg = None
    if (
        basename != mriot.basename
        or year != mriot.year
        or mriot.sectors_agg != sectors_agg
        or mriot.regions_agg != regions_agg
    ):
        if set_attribute_from_filename:
            mriot.basename = basename
            mriot.year = year
            mriot.sectors_agg = sectors_agg
            mriot.regions_agg = regions_agg
        else:
            warnings.warn(
                f"""Attribute and file name differ, you might want to look into that, or load with "set_attribute_from_filename=True":
        basename: {basename}, attribute: {mriot.basename}
        year: {year}, attribute: {mriot.year}
        sectors_agg: {sectors_agg}, attribute: {mriot.sectors_agg}
        regions_agg: {regions_agg}, attribute: {mriot.regions_agg}
        """
            )

    if not isinstance(mriot, pymrio.IOSystem):
        raise ValueError(f"{filename} was loaded but it is not an IOSystem")
    return mriot


######################################################################################

#### Reformat, reindex, apply corrections


def euregio_correct_regions(euregio: pymrio.IOSystem):
    euregio.rename_regions(EUREGIO_REGIONS_RENAMING).aggregate_duplicates()
    return euregio


def lexico_reindex(mriot: pymrio.IOSystem) -> pymrio.IOSystem:
    """Re-index IOSystem lexicographically.

    Sort indexes and columns of the dataframe of a pymrio.IOSystem by lexical order.

    Parameters
    ----------
    mriot : pymrio.IOSystem
        The IOSystem to sort.

    Returns
    -------
    pymrio.IOSystem
        The sorted IOSystem.
    """

    for matrix_name in ["Z", "Y", "x", "A", "As", "G", "L"]:
        matrix = getattr(mriot, matrix_name)
        if matrix is not None:
            setattr(
                mriot, matrix_name, matrix.reindex(sorted(matrix.index)).sort_index(axis=1)
            )

    return mriot


######################################################################################

### Compute stats / additional components


def va_df_build(
    mrios: dict, mrio_params: dict, mrio_unit: int = 10**6
) -> pd.DataFrame:
    """Builds a DataFrame containing value-added indicators from a set of pymrio.IOSystem objects. Negative value-added is set to 0.

    Parameters
    ----------
    mrios : dict
        A dictionary containing pymrio.IOSystem objects for different years.
    mrio_params : dict
        A dictionary containing MRIOT system parameters, including capital_ratio_dict.
    mrio_unit : int, optional
        A multiplier for monetary values (default is 10^6).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing value-added indicators for different years and regions.

    Raises
    ------
    AttributeError
        If required members (x, Z, Y) of pymrio.IOSystem are not set in the input MRIO systems.

    Notes
    -----
    The function computes various value-added indicators, including gross value added (GVA),
    capital stock, GVA share, yearly gross output, and yearly total final demand.

    """
    va_dict = {}

    for year, mrio in mrios.items():  # Use items() to iterate over key, value pairs
        mrio = lexico_reindex(mrio)

        if mrio.x is None or mrio.Z is None:
            raise AttributeError("x and Z members of mriot are not set.")

        value_added = mrio.x.T - mrio.Z.sum(axis=0)
        value_added = value_added.reindex(sorted(value_added.index), axis=0)
        value_added = value_added.reindex(sorted(value_added.columns), axis=1)
        value_added[value_added < 0] = 0.0
        va = value_added.T
        va = va.rename(columns={"indout": "GVA (M€)"})
        va["GVA (€)"] = va["GVA (M€)"] * mrio_unit

        va[["K_stock (M€)", "K_stock (€)"]] = va.loc[
            va.index.isin(list(mrio_params["capital_ratio_dict"].keys()), level=1), :
        ].mul(pd.Series(mrio_params["capital_ratio_dict"]), axis=0, level=1)

        va["gva_share"] = va["GVA (€)"] / va.groupby("region")["GVA (€)"].transform(
            "sum"
        )
        va["yearly gross output (M€)"] = mrio.x["indout"]
        va["yearly gross output (€)"] = mrio.x["indout"] * mrio_unit

        if mrio.Y is None:
            raise AttributeError("Y member of mrio is not set.")

        va["yearly total final demand (M€)"] = mrio.Y.sum(axis=1)
        va["yearly total final demand (€)"] = (
            va["yearly total final demand (M€)"] * mrio_unit
        )

        va = va.reset_index()
        LOGGER.info(f"year: {year}")
        va_dict[year] = va.set_index(["region", "sector"])

    va_df = pd.concat(va_dict.values(), axis=1, keys=va_dict.keys())
    va_df.columns = va_df.columns.rename("MRIOT year", level=0)  # type: ignore
    return va_df


def build_impacted_kstock_df(va_df, event_template):
    return (
        va_df.loc[
            pd.IndexSlice[:, event_template["aff_sectors"]],
            pd.IndexSlice[:, "K_stock (€)"],
        ]
    ).droplevel(1, axis=1)


def build_impacted_shares_df(va_df, event_template):
    return (
        va_df.loc[
            pd.IndexSlice[:, event_template["aff_sectors"]],
            pd.IndexSlice[:, "gva_share"],
        ]
        / va_df.loc[
            pd.IndexSlice[:, event_template["aff_sectors"]],
            pd.IndexSlice[:, "gva_share"],
        ]
        .groupby("region")
        .transform(sum)
    ).droplevel(1, axis=1)


######################################################################################


### Aggregation
def find_sectors_agg(basename, orig_agg, to_agg, agg_files_path):
    if to_agg == "common_sectors":
        agg_file = Path(agg_files_path) / "sectors_common_aggreg.ods"
        LOGGER.info("Reading aggregation from {}".format(agg_file.absolute()))
        return pd.read_excel(
            agg_file,
            sheet_name=f"{basename}_{orig_agg}_to_common_aggreg",
            index_col=0,
        )
    else:
        agg_file = (
            Path(agg_files_path) / basename / f"{basename}_{to_agg}.csv"
        )
        LOGGER.info("Reading aggregation from {}".format(agg_file.absolute()))
        return pd.read_csv(agg_file, index_col=0)


def find_regions_agg(basename, orig_agg, to_agg, agg_files_path):
    if to_agg == "common_regions":
        agg_file = Path(agg_files_path) / "regions_common_aggreg.ods"
        LOGGER.info("Reading aggregation from {}".format(agg_file.absolute()))
        return pd.read_excel(
            agg_file,
            sheet_name=f"{basename}_{orig_agg}_to_common_aggreg",
            index_col=0,
        )
    else:
        agg_file = (
            Path(agg_files_path) / basename / f"{basename}_{to_agg}.csv"
        )
        LOGGER.info("Reading aggregation from {}".format(agg_file.absolute()))
        return pd.read_csv(agg_file, index_col=0)

def aggreg(
    mriot: pymrio.IOSystem,
    sectors_aggregation=None,
    regions_aggregation=None,
    save_dir=None,
    agg_files_path=None,
):
    mriot = copy.deepcopy(mriot)
    assert isinstance(mriot, pymrio.IOSystem)

    with resources.path("boario_tools.data", "aggregation_files") as agg_files_path:
        if sectors_aggregation is not None:
            reg_agg_vec = find_sectors_agg(mriot.basename, mriot.sectors_agg, sectors_aggregation, agg_files_path)
            reg_agg_vec.sort_index(inplace=True)
            LOGGER.info(
                "Aggregating from {} to {} sectors".format(
                    mriot.get_sectors().nunique(), len(reg_agg_vec["new sector"].unique())  # type: ignore
                )
            )  # type:ignore
            mriot.rename_sectors(reg_agg_vec["new sector"].to_dict())
            mriot.aggregate_duplicates()
            mriot.sectors_agg = sectors_aggregation

        if regions_aggregation is not None:
            reg_agg_vec = find_regions_agg(mriot.basename, mriot.regions_agg, regions_aggregation, agg_files_path)
            reg_agg_vec.sort_index(inplace=True)
            LOGGER.info(
                "Aggregating from {} to {} regions".format(
                    mriot.get_regions().nunique(), len(reg_agg_vec["new region"].unique())  # type: ignore
                )
            )  # type:ignore
            mriot.rename_regions(reg_agg_vec["new region"].to_dict())
            mriot.aggregate_duplicates()
            mriot.regions_agg = regions_aggregation

    mriot.calc_all()
    mriot = lexico_reindex(mriot)
    LOGGER.info("Done")
    if save_dir:
        savefile = f"{save_dir}/{mriot.basename}_{mriot.year}_{mriot.sectors_agg}_{mriot.regions_agg}.pkl"
        LOGGER.info(f"Saving to {savefile}")
        with open(str(savefile), "wb") as f:
            pkl.dump(mriot, f)
    return mriot


######################################################################################

def check_sectors_in_mriot(sectors: Iterable[str], mriot: pymrio.IOSystem) -> None:
    """
    Check whether the given list of sectors exists within the MRIOT data.

    Parameters
    ----------
    sectors : list of str
        List of sector names to check.
    mriot : pym.IOSystem
        An instance of `pymrio.IOSystem`, representing the multi-regional input-output model.

    Raises
    ------
    ValueError
        If any of the sectors in the list are not found in the MRIOT data.
    """
    # Retrieve all available sectors from the MRIOT data
    available_sectors = set(mriot.get_sectors())

    # Identify missing sectors
    missing_sectors = set(sectors) - available_sectors

    # Raise an error if any sectors are missing
    if missing_sectors:
        raise ValueError(
            f"The following sectors are missing in the MRIOT data: {missing_sectors}"
        )

def get_coco_MRIOT_name(mriot_name):
    match = MRIOT_FULLNAME_REGEX.match(mriot_name)
    if not match:
        raise ValueError(f"Input string '{mriot_name}' is not in the correct format '<MRIOT-name>_<year>' or not recognized.")
    mriot_type = match.group("mrio_type")
    return MRIOT_COUNTRY_CONVERTER_CORR[mriot_type]
