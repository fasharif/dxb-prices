"""Audit the cached raw export for the facts docs/data.md and the model card rely on.

Prints, for the cached raw files:
* Arabic community names that carry more than one English spelling,
* English community names that carry more than one Arabic spelling,
* how the apartment sale procedures split between freehold and non-freehold,
* which groups share a transaction number (lease-to-own contracts),
* how often the optional columns are empty,
* the most expensive apartment sales per square metre that pass the cleaning rules.

Usage: python scripts/data_audit.py [RAW_DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from dxb_prices import clean, config, normalise, schema


def names(df: pd.DataFrame) -> None:
    keys = pd.DataFrame(
        {
            "en": df["community_en"].map(normalise.key_en, na_action="ignore"),
            "ar": df["community_ar"].map(normalise.key_ar, na_action="ignore"),
            "community_en": df["community_en"],
        }
    ).dropna(subset=["en", "ar"])
    pairs = keys.drop_duplicates()
    multi_en = pairs.groupby("ar")["en"].nunique()
    print(f"Arabic names with more than one English name: {int((multi_en > 1).sum())}")
    for ar in multi_en[multi_en > 1].index:
        spellings = sorted(pairs.loc[pairs["ar"] == ar, "community_en"].unique())
        print(f"  {ar}: {' | '.join(spellings)}")
    multi_ar = pairs.groupby("en")["ar"].nunique()
    print(f"English names with more than one Arabic name: {int((multi_ar > 1).sum())}")


def procedures(df: pd.DataFrame) -> None:
    sub_types = config.DEFAULT_SETTINGS.cleaning.residential_sub_types
    sales = df[
        df["group"].str.casefold().eq("sales").fillna(False)
        & df["property_sub_type"].isin(sub_types)
    ]
    freehold = sales["is_freehold"].map({True: "freehold", False: "non-freehold"})
    table = pd.crosstab(sales["procedure"], freehold.fillna("unknown"))
    print("\nApartment sale procedures by freehold flag:")
    print(table.to_string())


def shared_numbers(df: pd.DataFrame) -> None:
    unique = df.drop_duplicates()
    repeated = unique[unique["transaction_id"].duplicated(keep=False)]
    groups = repeated.groupby("transaction_id")["group"].agg(lambda s: "+".join(sorted(set(s))))
    print("\nTransaction numbers on several rows, by the groups involved:")
    print(groups.value_counts().to_string())
    both = repeated[repeated["transaction_id"].isin(groups[groups == "Mortgage+Sales"].index)]
    print("Procedures of numbers shared by Sales and Mortgage:")
    print(both["procedure"].value_counts().to_string())


def empty_columns(df: pd.DataFrame) -> None:
    cleaned, _ = clean.clean(df, config.DEFAULT_SETTINGS.cleaning)
    ids = set(cleaned["transaction_id"])
    kept = df[df["transaction_id"].isin(ids)].drop_duplicates("transaction_id")
    print("\nShare empty among cleaned apartment sales:")
    for col in (
        "master_project_en",
        "project_en",
        "nearest_metro",
        "nearest_mall",
        "nearest_landmark",
    ):
        print(f"  {col}: {kept[col].isna().mean():.1%}")


def luxury(df: pd.DataFrame) -> None:
    cleaned, _ = clean.clean(df, config.DEFAULT_SETTINGS.cleaning)
    pps = cleaned["price_aed"] / cleaned["area_sqm"]
    print("\nCleaned apartment sales:", f"{len(cleaned):,}")
    print("Highest price per sqm (AED):", f"{pps.max():,.0f}")
    print("Sales above 150,000 AED per sqm:", int((pps > 150_000).sum()))
    print("Largest size (sqm):", f"{cleaned['area_sqm'].max():,.0f}")


def main(raw_dir: Path) -> None:
    raw = schema.read_raw_csvs(sorted(raw_dir.glob("transactions_*.csv")))
    df = schema.to_canonical(raw)
    print(f"Raw rows: {len(df):,}")
    names(df)
    procedures(df)
    shared_numbers(df)
    empty_columns(df)
    luxury(df)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else config.RAW_DIR)
