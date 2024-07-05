import os
from pathlib import Path
import re
from typing import Union
import numpy as np
import pandas as pd
import pymrio
import pickle as pkl
from boario_tools import log
import copy

import warnings
from importlib import resources
from boario_tools.regex_patterns import MRIOT_FULLNAME_REGEX, MRIOT_YEAR_REGEX

POSSIBLE_MRIOT_REGEXP = MRIOT_FULLNAME_REGEX
EUREGIO_REGIONS_RENAMING = {"DEE1": "DEE0", "DEE2": "DEE0", "DEE3": "DEE0"}

ATTR_LIST = [
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


def conda_check():
    log.info(
        """Make sure you use the same python environment as the one loading
        the pickle file (especial pymrio and pandas version !)"""
    )
    try:
        log.info("Your current environment is: {}".format(os.environ["CONDA_PREFIX"]))
    except KeyError:
        log.info(
            "Could not find CONDA_PREFIX, this is normal if you are not using conda."
        )


####################### Downloading and parsing ######################################
def build_exio3_from_zip(mrio_zip: str, remove_attributes):
    conda_check()
    mrio_path = Path(mrio_zip)
    mrio_pym = pymrio.parse_exiobase3(path=mrio_path)
    if remove_attributes:
        log.info("Removing unnecessary IOSystem attributes")
        attr = ATTR_LIST
        tmp = list(mrio_pym.__dict__.keys())
        for at in tmp:
            if at not in attr:
                delattr(mrio_pym, at)
    assert isinstance(mrio_pym, pymrio.IOSystem)
    log.info("Done")
    log.info("Computing the missing IO components")
    mrio_pym.calc_all()
    log.info("Done")
    log.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    log.info("Done")
    setattr(mrio_pym, "monetary_factor", 1000000)
    setattr(mrio_pym, "basename", "exiobase3_ixi")
    setattr(mrio_pym, "year", mrio_pym.meta.description[-4:])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    return mrio_pym


def euregio_convert_xlsx2csv(inpt, out_folder, office_exists):
    if not office_exists:
        raise FileNotFoundError(
            "Creating csvs files require libreoffice which wasn't found. You may wan't to convert EUREGIO files by yourself if you are unable to install libreoffice"
        )
    log.info(
        f"Executing: libreoffice --convert-to 'csv:Text - txt - csv (StarCalc):44,34,0,1,,,,,,,,3' {out_folder} {inpt}"
    )
    os.system(
        f"libreoffice --convert-to 'csv:Text - txt - csv (StarCalc):44,34,0,1,,,,,,,,3' --outdir {out_folder} {inpt}"
    )
    filename = Path(inpt).name
    log.info(filename)
    new_filename = (
        "euregio_" + filename.split("_")[1].split(".")[0].replace("-", "_") + ".csv"
    )
    old_path = Path(out_folder) / filename.replace(
        ".xlsb", "-{}.csv".format(filename.split("_")[1].split(".")[0])
    )
    new_path = Path(out_folder) / new_filename
    log.info(f"Executing: mv {old_path} {new_path}")
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
        log.info(f"Correcting germany regions : {EUREGIO_REGIONS_RENAMING}")
        euregio = euregio_correct_regions(euregio)
    log.info("Computing the missing IO components")
    euregio.calc_all()
    euregio.meta.change_meta("name", f"euregio {year}")

    assert isinstance(euregio, pymrio.IOSystem)
    log.info("Re-indexing lexicographicaly")
    euregio = lexico_reindex(euregio)
    log.info("Done")
    return euregio


def build_eora_from_zip(
    mrio_zip: str,
    reexport_treatment,
    inv_treatment,
    remove_attributes,
):
    conda_check()
    mrio_path = Path(mrio_zip)
    mrio_pym = pymrio.parse_eora26(path=mrio_path)
    log.info("Removing unnecessary IOSystem attributes")
    if remove_attributes:
        attr = ATTR_LIST
        tmp = list(mrio_pym.__dict__.keys())
        for at in tmp:
            if at not in attr:
                delattr(mrio_pym, at)
    assert isinstance(mrio_pym, pymrio.IOSystem)
    log.info("Done")
    setattr(mrio_pym, "monetary_factor", 1000)
    setattr(mrio_pym, "basename", "eora26")
    setattr(mrio_pym, "year", re.search(MRIOT_YEAR_REGEX, mrio_zip)["mrio_year"])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")

    if reexport_treatment:
        log.info(
            "EORA26 has the re-import/re-export sector which other mrio often don't have (ie EXIOBASE), we put it in 'Other'."
        )
        mrio_pym.rename_sectors({"Re-export & Re-import": "Others"})
        mrio_pym.aggregate_duplicates()
        setattr(mrio_pym, "sectors_agg", "full_no_reexport_sectors")

    if inv_treatment:
        log.info(
            "EORA26 has negative values in its final demand which can cause problems. We set them to 0."
        )
        if mrio_pym.Y is not None:
            mrio_pym.Y = mrio_pym.Y.clip(lower=0)
        else:
            raise AttributeError("Y attribute is not set")
    log.info("Computing the missing IO components")
    mrio_pym.calc_all()
    log.info("Done")
    log.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    log.info("Done")
    return mrio_pym


def build_oecd_from_zip(mrio_zip: str, year: int):
    conda_check()
    mrio_path = Path(mrio_zip)
    mrio_pym = pymrio.parse_oecd(path=mrio_path, year=year)
    log.info("Removing unnecessary IOSystem attributes")
    attr = ATTR_list
    tmp = list(mrio_pym.__dict__.keys())
    for at in tmp:
        if at not in attr:
            delattr(mrio_pym, at)
    assert isinstance(mrio_pym, pymrio.IOSystem)
    log.info("Done")
    setattr(mrio_pym, "monetary_factor", 1000000)
    setattr(mrio_pym, "basename", "icio_v2018")
    setattr(mrio_pym, "year", year)
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    log.info("Computing the missing IO components")
    mrio_pym.calc_all()
    log.info("Done")
    log.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    log.info("Done")


def parse_mriot_from_df(
    mriot_df: pd.DataFrame,
    col_iso3: int,
    col_sectors: int,
    rows_data: tuple[int, int],
    cols_data: tuple[int, int],
    row_fd_cats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build multi-index dataframes of the transaction matrix, final demand and total
       production from a Multi-Regional Input-Output Table dataframe.

    - Adapted from Alessio Ciulio's contribution to Climada -

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
        containing the MRIOT data
    cols_data : (int, int)
        Tuple of integers with positions of columns
        containing the MRIOT data
    row_fd_cats : int
        Row's position of final demand categories
    """

    start_row, end_row = rows_data
    start_col, end_col = cols_data

    sectors = mriot_df.iloc[start_row:end_row, col_sectors].unique()
    regions = mriot_df.iloc[start_row:end_row, col_iso3].unique()
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
    x = pd.DataFrame(data=x, index=multiindex, columns=["total production"])

    return Z, Y, x


def parse_wiod_v2016(mrio_xlsb: str):
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
        data=np.repeat(["M.EUR"], len(multiindex_unit)),
        index=multiindex_unit,
        columns=["unit"],
    )
    setattr(mrio_pym, "monetary_factor", 1000000)
    setattr(mrio_pym, "basename", "wiod_v2016")
    setattr(mrio_pym, "year", None)
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")

    assert isinstance(mrio_pym, pymrio.IOSystem)
    log.info("Computing the missing IO components")
    mrio_pym.calc_all()
    log.info("Done")
    log.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    log.info("Done")
    return mrio_pym


######################################################################################


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
    log.info("Saving to {}".format(save_path.absolute()))
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
    log.info("Saving to {}".format(save_path.absolute()))
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
    log.info("Saving to {}".format(save_path.absolute()))
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
    log.info("Saving to {}".format(save_path.absolute()))
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
    log.info("Saving to {}".format(save_path.absolute()))
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

    log.info(f"Loading {filename} mrio")
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
    for attr in ["Z", "Y", "x", "A"]:
        if getattr(mriot, attr) is None:
            raise ValueError(
                f"Attribute {attr} is None. Did you forget to calc_all() the MRIOT?"
            )

    for matrix_name in ["Z", "Y", "x", "A"]:
        matrix = getattr(mriot, matrix_name)
        setattr(
            mriot, matrix_name, matrix.reindex(sorted(matrix.index)).sort_index(axis=1)
        )

    return mriot


def euregio_correct_regions(euregio: pymrio.IOSystem):
    euregio.rename_regions(EUREGIO_REGIONS_RENAMING).aggregate_duplicates()
    return euregio


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
        log.info(f"year: {year}")
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
        log.info("Reading aggregation from {}".format(agg_file.absolute()))
        return pd.read_excel(
            agg_file,
            sheet_name=f"{basename}_{orig_agg}_to_common_aggreg",
            index_col=0,
        )
    else:
        agg_file = (
            Path(agg_files_path) / basename / f"{basename}_{to_agg}.csv"
        )
        log.info("Reading aggregation from {}".format(agg_file.absolute()))
        return pd.read_csv(agg_file, index_col=0)


def find_regions_agg(basename, orig_agg, to_agg, agg_files_path):
    if to_agg == "common_regions":
        agg_file = Path(agg_files_path) / "regions_common_aggreg.ods"
        log.info("Reading aggregation from {}".format(agg_file.absolute()))
        return pd.read_excel(
            agg_file,
            sheet_name=f"{basename}_{orig_agg}_to_common_aggreg",
            index_col=0,
        )
    else:
        agg_file = (
            Path(agg_files_path) / basename / f"{basename}_{to_agg}.csv"
        )
        log.info("Reading aggregation from {}".format(agg_file.absolute()))
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
            log.info(
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
            log.info(
                "Aggregating from {} to {} regions".format(
                    mriot.get_regions().nunique(), len(reg_agg_vec["new region"].unique())  # type: ignore
                )
            )  # type:ignore
            mriot.rename_regions(reg_agg_vec["new region"].to_dict())
            mriot.aggregate_duplicates()
            mriot.regions_agg = regions_aggregation

    mriot.calc_all()
    mriot = lexico_reindex(mriot)
    log.info("Done")
    if save_dir:
        savefile = f"{save_dir}/{mriot.basename}_{mriot.year}_{mriot.sectors_agg}_{mriot.regions_agg}.pkl"
        log.info(f"Saving to {savefile}")
        with open(str(savefile), "wb") as f:
            pkl.dump(mriot, f)
    return mriot


######################################################################################
