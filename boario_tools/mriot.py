import os
from pathlib import Path
import re
from typing import Union
import numpy as np
import pandas as pd
import pymrio
import pickle as pkl
from boario_tools import log

POSSIBLE_MRIOT_REGEXP = r"^(oecd_v2021|euregio|exiobase3|eora26)_full_(\d{4})"
EUREGIO_REGIONS_RENAMING = {"DEE1": "DEE0", "DEE2": "DEE0", "DEE3": "DEE0"}


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


def load_sectors_aggreg(mrio_name, sectors_common_aggreg):
    mrio_name = mrio_name.casefold()
    if "eora" in mrio_name:
        return sectors_common_aggreg["eora26_without_reexport_to_common_aggreg"]
    elif "euregio" in mrio_name:
        return sectors_common_aggreg["euregio_to_common_aggreg"]
    elif "exio" in mrio_name:
        return sectors_common_aggreg["exiobase_full_to_common_aggreg"]
    elif "oecd" in mrio_name:
        return sectors_common_aggreg["icio2021_to_common_aggreg"]
    else:
        raise ValueError(f"Invalid MRIO name: {mrio_name}")


def common_aggreg(sector_agg, mrio):
    sectors_common_aggreg = {
        sheet_name: pd.read_excel(sector_agg, sheet_name=sheet_name, index_col=0)
        for sheet_name in [
            "eora26_without_reexport_to_common_aggreg",
            "euregio_to_common_aggreg",
            "exiobase_full_to_common_aggreg",
            "icio2021_to_common_aggreg",
        ]
    }
    df_aggreg = load_sectors_aggreg(mrio.name, sectors_common_aggreg)
    mrio.rename_sectors(df_aggreg["new sector"].to_dict())
    mrio.aggregate_duplicates()
    return mrio


def aggreg(
    mrio_path: Union[str, Path],
    sector_aggregator_path: Union[str, Path],
    save_path=None,
):
    log.info("Loading sector aggregator")
    log.info(
        "Make sure you use the same python environment as the one loading the pickle file (especial pymrio and pandas version !)"
    )
    log.info("Your current environment is: {}".format(os.environ["CONDA_PREFIX"]))

    mrio_path = Path(mrio_path)
    if not mrio_path.exists():
        raise FileNotFoundError("MRIO file not found - {}".format(mrio_path))

    if mrio_path.suffix == ".pkl":
        with mrio_path.open("rb") as f:
            log.info("Loading MRIO from {}".format(mrio_path.resolve()))
            mrio = pkl.load(f)
    else:
        raise TypeError(
            "File type ({}) not recognize for the script (must be zip or pkl) : {}".format(
                mrio_path.suffix, mrio_path.resolve()
            )
        )

    assert isinstance(mrio, pymryo.IOSystem)
    log.info(
        "Reading aggregation from {}".format(Path(sector_aggregator_path).absolute())
    )

    if "common_aggreg" in str(sector_aggregator_path):
        mrio = common_aggreg(sector_aggregator_path, mrio)
    else:
        sec_agg_vec = pd.read_csv(sector_aggregator_path, index_col=0)
        sec_agg_vec.sort_index(inplace=True)

        log.info(
            "Aggregating from {} to {} sectors".format(
                mrio.get_sectors().nunique(), len(sec_agg_vec.group.unique())  # type: ignore
            )
        )  # type:ignore
        mrio.aggregate(sector_agg=sec_agg_vec.name.values)

    mrio.calc_all()
    log.info("Done")
    log.info(f"Saving to {save_path}")
    with open(str(save_path), "wb") as f:
        pkl.dump(mrio, f)


def load_mrio(filename: str, pkl_filepath) -> pymrio.IOSystem:
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
        POSSIBLE_MRIOT_REGEXP
    )  # the regular expression to match filenames

    match = regex.match(filename)  # match the filename with the regular expression

    if not match:
        raise ValueError(f"The file name {filename} is not valid.")

    prefix, _ = match.groups()  # get the prefix and year from the matched groups

    pkl_filepath = Path(pkl_filepath)

    fullpath = pkl_filepath / prefix / f"{filename}.pkl"  # create the full file path

    log.info(f"Loading {filename} mrio")
    with open(fullpath, "rb") as f:
        mrio = pkl.load(f)  # load the pickle file

    if not isinstance(mrio, pymrio.IOSystem):
        raise ValueError(f"{filename} was loaded but it is not an IOSystem")

    return mrio


def preparse_exio3(mrio_zip: str, output: str):
    log.info(
        "Make sure you use the same python environment as the one loading the pickle file (especial pymrio and pandas version !)"
    )
    log.info("Your current environment is: {}".format(os.environ["CONDA_PREFIX"]))
    mrio_path = Path(mrio_zip)
    mrio_pym = pymrio.parse_exiobase3(path=mrio_path)
    log.info("Removing unnecessary IOSystem attributes")
    attr = [
        "Z",
        "Y",
        "x",
        "A",
        "L",
        "unit",
        "population",
        "meta",
        "__non_agg_attributes__",
        "__coefficients__",
        "__basic__",
    ]
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
    save_path = Path(output)
    log.info("Saving to {}".format(save_path.absolute()))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    setattr(mrio_pym, "monetary_factor", 1000000)
    with open(save_path, "wb") as f:
        pkl.dump(mrio_pym, f)


def euregio_convert_ods2xlsx(inpt, output, folder, office_exists, uno_exists):
    if not office_exists:
        raise FileNotFoundError(
            "Creating xlsx files require at least libreoffice which wasn't found (and optionally unoserver). You may wan't to convert EUREGIO files by yourself if you are unable to install libreoffice"
        )
    if uno_exists:
        os.system(f"unoconvert --port 2002 --convert-to xlsx {inpt} {output}")
    else:
        os.system(f"libreoffice --convert-to xlsx --outdir {folder} {inpt}")


def euregio_correct_regions(euregio: pymrio.IOSystem):
    euregio.rename_regions(EUREGIO_REGIONS_RENAMING).aggregate_duplicates()
    return euregio


def build_from_csv(csv_file, inv_treatment=True):
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

    # ioz = ioz.rename_axis(["region","sector"])
    # ioz = ioz.rename_axis(["region","sector"],axis=1)

    ioy = ioy.rename_axis(["region", "sector"])
    ioy = ioy.rename_axis(["region", "category"], axis=1)
    if inv_treatment:
        # invs = ioy.loc[:, (slice(None), "Inventory_adjustment")].sum(axis=1)
        # invs.name = "Inventory_use"
        # invs_neg = pd.DataFrame(-invs).T
        # invs_neg[invs_neg < 0] = 0
        # iova = pd.concat([iova, invs_neg], axis=0)
        ioy = ioy.clip(lower=0)

    return ioz, ioy, iova


def preparse_euregio(mrio_csv: str, output: str, year):
    log.info(
        "Make sure you use the same python environment as the one loading the pickle file (especial pymrio and pandas version !)"
    )
    log.info("Your current environment is: {}".format(os.environ["CONDA_PREFIX"]))
    ioz, ioy, iova = build_from_csv(mrio_csv, inv_treatment=True)
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
    log.info(f"Correcting germany regions : {EUREGIO_REGIONS_RENAMING}")
    euregio = euregio_correct_regions(euregio)
    log.info("Computing the missing IO components")
    euregio.calc_all()
    euregio.meta.change_meta("name", f"euregio {year}")

    assert isinstance(euregio, pymrio.IOSystem)
    log.info("Re-indexing lexicographicaly")
    euregio = lexico_reindex(euregio)
    log.info("Done")
    save_path = Path(output)
    log.info("Saving to {}".format(save_path.absolute()))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    setattr(euregio, "monetary_factor", 1000000)
    with open(save_path, "wb") as f:
        pkl.dump(euregio, f)


def preparse_eora26(mrio_zip: str, output: str, inv_treatment=True):
    log.info(
        "Make sure you use the same python environment as the one loading the pickle file (especial pymrio and pandas version !)"
    )
    log.info("Your current environment is: {}".format(os.environ["CONDA_PREFIX"]))
    mrio_path = Path(mrio_zip)
    mrio_pym = pymrio.parse_eora26(path=mrio_path)
    log.info("Removing unnecessary IOSystem attributes")
    attr = [
        "Z",
        "Y",
        "x",
        "A",
        "L",
        "unit",
        "population",
        "meta",
        "__non_agg_attributes__",
        "__coefficients__",
        "__basic__",
    ]
    tmp = list(mrio_pym.__dict__.keys())
    for at in tmp:
        if at not in attr:
            delattr(mrio_pym, at)
    assert isinstance(mrio_pym, pymrio.IOSystem)
    log.info("Done")
    log.info(
        'EORA has the re-import/re-export sector which other mrio often don\'t have (ie EXIOBASE), we put it in "Other".'
    )
    mrio_pym.rename_sectors({"Re-export & Re-import": "Others"})
    mrio_pym.aggregate_duplicates()

    if inv_treatment:
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
    save_path = Path(output)
    log.info("Saving to {}".format(save_path.absolute()))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    setattr(mrio_pym, "monetary_factor", 1000)
    with open(save_path, "wb") as f:
        pkl.dump(mrio_pym, f)


def preparse_oecd_v2018(mrio_zip: str, output: str):
    log.info(
        "Make sure you use the same python environment as the one loading the pickle file (especial pymrio and pandas version !)"
    )
    log.info("Your current environment is: {}".format(os.environ["CONDA_PREFIX"]))
    mrio_path = Path(mrio_zip)
    mrio_pym = pymrio.parse_oecd(path=mrio_path)
    log.info("Removing unnecessary IOSystem attributes")
    attr = [
        "Z",
        "Y",
        "x",
        "A",
        "L",
        "unit",
        "population",
        "meta",
        "__non_agg_attributes__",
        "__coefficients__",
        "__basic__",
    ]
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
    save_path = Path(output)
    log.info("Saving to {}".format(save_path.absolute()))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    setattr(mrio_pym, "monetary_factor", 1000000)
    with open(save_path, "wb") as f:
        pkl.dump(mrio_pym, f)


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


def parse_wiod_v2016(mrio_xlsb: str, output: str):
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

    assert isinstance(mrio_pym, pymrio.IOSystem)
    log.info("Computing the missing IO components")
    mrio_pym.calc_all()
    log.info("Done")
    log.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    log.info("Done")
    save_path = Path(output)
    log.info("Saving to {}".format(save_path.absolute()))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    setattr(mrio_pym, "monetary_factor", 1000000)
    with open(save_path, "wb") as f:
        pkl.dump(mrio_pym, f)
