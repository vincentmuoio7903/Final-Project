
#!/usr/bin/env python3
"""
data_wrangling.py
Utility for loading, cleaning, and exporting LA crime data (2020‑present).

Usage (CLI):
    python data_wrangling.py /path/to/Crime_Data_from_2020_to_Present.csv -o cleaned_crime_data.parquet

The script:
  • Reads the raw CSV, parses date columns, and combines date+time into a single `occurrence_datetime`.
  • Renames columns to snake_case for consistency.
  • Drops duplicate rows and those missing essential geocoordinates.
  • Keeps a focused subset of informative columns.
  • Writes out a cleaned CSV or Parquet file, depending on the `-o/--out` flag.

Functions
---------
load_and_clean(filepath: str) -> pd.DataFrame
    Returns a cleaned DataFrame so you can import the function in notebooks.

main()
    Command‑line entry point. Handles argument parsing and file export.

"""

import argparse
from pathlib import Path

import pandas as pd


def _to_snake(name: str) -> str:
    """Convert column names to snake_case."""
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load raw LA crime CSV and return a cleaned DataFrame."""
    df = pd.read_csv(
        filepath,
        parse_dates=["Date Rptd", "DATE OCC"],
        low_memory=False,
    )

    # Rename columns to snake_case
    df.columns = [_to_snake(c) for c in df.columns]

    # Standardize TIME_OCC – ensure four‑digit zero‑padded strings (HHMM)
    if "time_occ" in df.columns:
        df["time_occ"] = (
            df["time_occ"]
            .fillna(0)
            .astype(int)
            .apply(lambda x: f"{x:04d}")
        )

        # Combine DATE_OCC and TIME_OCC into a single timestamp
        df["occurrence_datetime"] = pd.to_datetime(
            df["date_occ"].dt.strftime("%Y-%m-%d")
            + " "
            + df["time_occ"].str[:2]
            + ":"
            + df["time_occ"].str[2:],
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Drop rows missing latitude/longitude (if present)
    lat_cols = [c for c in df.columns if c.startswith("lat")]
    lon_cols = [c for c in df.columns if c.startswith(("lon", "lng"))]
    essential_cols = lat_cols + lon_cols
    if essential_cols:
        df.dropna(subset=essential_cols, inplace=True)

    # Select a subset of useful columns if they exist
    candidate_cols = [
        "dr_no",
        "date_rptd",
        "occurrence_datetime",
        "crm_cd_desc",
        "vict_age",
        "vict_sex",
        *essential_cols,
    ]
    keep_cols = [c for c in candidate_cols if c in df.columns]
    df_clean = df[keep_cols].copy()

    return df_clean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean raw LA crime data CSV and save as CSV or Parquet."
    )
    parser.add_argument("input_csv", help="Path to raw CSV file")
    parser.add_argument(
        "-o",
        "--out",
        default="cleaned_crime_data.parquet",
        help="Output filename (.csv or .parquet)",
    )
    args = parser.parse_args()

    cleaned = load_and_clean(args.input_csv)
    out_path = Path(args.out)

    if out_path.suffix.lower() == ".csv":
        cleaned.to_csv(out_path, index=False)
    else:
        cleaned.to_parquet(out_path, index=False)

    print(f"✔ Saved {len(cleaned):,} cleaned rows to {out_path}")
    print("Columns:", ", ".join(cleaned.columns))


if __name__ == "__main__":
    main()
