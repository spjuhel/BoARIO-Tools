import re

ORDER_TYPE_REGEX = r"(?P<order>alt|noalt)"
PSI_REGEX = r"(?P<psi>(?:1|0)\.\d+)"
BASE_ALPHA_REGEX = r"(?P<base_alpha>\d(?:\.\d+)?)"
MAX_ALPHA_REGEX = r"(?P<max_alpha>\d(?:\.\d+)?)"
TAU_ALPHA_REGEX = r"(?P<tau_alpha>\d+)"

MRIOT_BASENAME_REGEX = r"(?P<mrio_basename>icio2021|euregio|exiobase3_ixi|eora26)"
MRIOT_YEAR_REGEX = r"(?P<mrio_year>\d{4})"
MRIOT_AGGREG_SECTORS_REGEX = r"(?P<mrio_aggreg_sectors>[a-zA-Z0-9_]+_sectors)"
MRIOT_AGGREG_REGIONS_REGEX = r"(?P<mrio_aggreg_regions>[a-zA-Z0-9_]+_regions)"
MRIOT_AGGREG_REGEX = (
    r"""{MRIOT_AGGREG_SECTORS_REGEX}_{MRIOT_AGGREG_REGIONS_REGEX}""".format(
        MRIOT_AGGREG_SECTORS_REGEX=MRIOT_AGGREG_SECTORS_REGEX,
        MRIOT_AGGREG_REGIONS_REGEX=MRIOT_AGGREG_REGIONS_REGEX,
    )
)

MRIOT_FULLNAME_REGEX = re.compile(
    r"""
{MRIOT_BASENAME_REGEX} # MRIOT basename
_ #
{MRIOT_YEAR_REGEX} # MRIOT year
_ #
{MRIOT_AGGREG_REGEX} # Aggregation specification
""".format(
        MRIOT_BASENAME_REGEX=MRIOT_BASENAME_REGEX,
        MRIOT_YEAR_REGEX=MRIOT_YEAR_REGEX,
        MRIOT_AGGREG_REGEX=MRIOT_AGGREG_REGEX,
    ),
    re.VERBOSE,
)
