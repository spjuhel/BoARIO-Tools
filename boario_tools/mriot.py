import pandas as pd
from boario_tools import log

def lexico_reindex(mrio: pymrio.IOSystem) -> pymrio.IOSystem:
    mrio.Z = mrio.Z.reindex(sorted(mrio.Z.index), axis=0)
    mrio.Z = mrio.Z.reindex(sorted(mrio.Z.columns), axis=1)
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.index), axis=0)
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.columns), axis=1)
    mrio.x = mrio.x.reindex(sorted(mrio.x.index), axis=0)  # type: ignore
    mrio.A = mrio.A.reindex(sorted(mrio.A.index), axis=0)
    mrio.A = mrio.A.reindex(sorted(mrio.A.columns), axis=1)
    return mrio


def va_df_build(mrios, mrio_params: dict, mrio_unit: int = 10**6) -> pd.DataFrame:
    va_dict = {}
    for year, mrio in mrios:
        mrio = lexico_reindex(mrio)
        value_added = mrio.x.T - mrio.Z.sum(axis=0)
        value_added = value_added.reindex(sorted(value_added.index), axis=0)  # type: ignore
        value_added = value_added.reindex(sorted(value_added.columns), axis=1)
        value_added[value_added < 0] = 0.0
        va = value_added.T
        va = va.rename(columns={"indout": "GVA (M€)"})
        va["GVA (€)"] = va["GVA (M€)"] * mrio_unit
        # display(va)
        va[["K_stock (M€)", "K_stock (€)"]] = va.loc[
            va.index.isin(list(mrio_params["capital_ratio_dict"].keys()), level=1), :
        ].mul(pd.Series(mrio_params["capital_ratio_dict"]), axis=0, level=1)
        va["gva_share"] = va["GVA (€)"] / va.groupby("region")["GVA (€)"].transform(
            "sum"
        )
        va["yearly gross output (M€)"] = mrio.x["indout"]
        va["yearly gross output (€)"] = mrio.x["indout"] * mrio_unit
        va["yearly total final demand (M€)"] = mrio.Y.sum(axis=1)
        va["yearly total final demand (€)"] = (
            va["yearly total final demand (M€)"] * mrio_unit
        )
        va = va.reset_index()
        log.info(f"year: {year}")
        va_dict[year] = va.set_index(["region", "sector"])
    va_df = pd.concat(va_dict.values(), axis=1, keys=va_dict.keys())
    va_df.columns = va_df.columns.rename("MRIO year", level=0)
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
