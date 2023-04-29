from typing import Sequence
import logging
import argparse
from pathlib import Path

from floods import build_rep_events_from_parquet

from utils import (
    save_parquet_with_meta,
    file_path,
)

logFormatter = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S"
)
scriptLogger = logging.getLogger("Rep events builder")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
# scriptLogger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute representatives events from a database of events"
    )
    parser.add_argument(
        "-i",
        "--input-parquet",
        type=file_path,
        help="The path to the folder with Flood data",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-parquet",
        type=str,
        help="The path to the folder to save the data",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--column-base",
        type=str,
        help="The column to base the representativity on",
        required=True,
    )
    parser.add_argument(
        "-G",
        "--gva-max",
        type=float,
        help="maximum share of region gva damage",
        default=1.0,
    )
    args = parser.parse_args(argv)
    input_parquet = Path(args.input_parquet).resolve()
    output_parquet = Path(args.output_parquet).resolve()
    based_on_column = args.column_base
    gva_max = args.gva_max

    df = build_rep_events_from_parquet(
        input_parquet, based_on_column=based_on_column, gva_max=gva_max
    )
    save_parquet_with_meta(df=df, path=output_parquet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
