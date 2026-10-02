"""Audit the cached raw export for the facts docs/data.md and the model card rely on.

Prints, for the cached raw files:
* Arabic community names that carry more than one English spelling,
* English community names that carry more than one Arabic spelling,
* how the apartment sale procedures split between freehold and non-freehold,
* which groups share a transaction number (lease-to-own contracts),
* how often the optional columns are empty,
* the most expensive apartment sales per square metre that pass the cleaning rules,
* the sales in the newest month whose community had no sales in the earlier months,
* project names recorded in more than one community before the newest month,
* small communities whose every sale lies outside the overall 0.5% to 99.5% range of
  price per square metre (a trim against those percentiles would remove them entirely).

With ``--dubai-data-sample`` it instead checks the data.dubai public sample saved by
``dxb-prices download --source dubai-data-sample``: its size, its years, and what
the schema adapter and the cleaning rules make of the Dubai Pulse layout.

Usage: python scripts/data_audit.py [RAW_DIR] [--dubai-data-sample [CSV]]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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


def new_communities(df: pd.DataFrame) -> None:
    cleaned, _ = clean.clean(df, config.DEFAULT_SETTINGS.cleaning)
    newest = cleaned["month"].max()
    earlier = set(cleaned.loc[cleaned["month"] < newest, "community"])
    fresh = cleaned[(cleaned["month"] == newest) & ~cleaned["community"].isin(earlier)]
    print(f"\nSales in {newest} in communities with no earlier sales: {len(fresh)}")
    if not fresh.empty:
        print(f"  communities: {fresh['community'].nunique()}")
        low, high = fresh["price_aed"].min(), fresh["price_aed"].max()
        print(f"  price range (AED): {low:,.0f} to {high:,.0f}")
        print(f"  off-plan: {fresh['is_off_plan'].astype('boolean').mean():.0%}")


def shared_project_names(df: pd.DataFrame) -> None:
    cleaned, _ = clean.clean(df, config.DEFAULT_SETTINGS.cleaning)
    newest = cleaned["month"].max()
    earlier = cleaned[(cleaned["month"] < newest)].dropna(subset=["project"])
    per_name = earlier.groupby("project")["community"].nunique()
    shared = per_name[per_name > 1]
    print(
        f"\nProject names recorded in more than one community before {newest}: "
        f"{len(shared)} of {len(per_name):,}"
    )
    for name in shared.index:
        counts = earlier.loc[earlier["project"] == name, "community"].value_counts().sort_index()
        print(
            f"  {name}: "
            + ", ".join(f"{c} ({n} sale{'' if n == 1 else 's'})" for c, n in counts.items())
        )
    in_newest = cleaned[(cleaned["month"] == newest) & cleaned["project"].isin(shared.index)]
    print(f"  sales in {newest} with one of these names: {len(in_newest)}")


def small_outlying_communities(df: pd.DataFrame, min_rows: int = 200) -> None:
    cleaned, _ = clean.clean(df, config.DEFAULT_SETTINGS.cleaning)
    ratio = (cleaned["price_aed"] / cleaned["area_sqm"]).to_numpy(np.float64)
    log_pps = pd.Series(np.log(ratio), index=cleaned.index)
    trim = config.DEFAULT_SETTINGS.trim
    lo, hi = log_pps.quantile([trim.lower_quantile, trim.upper_quantile])
    counts = cleaned["community"].value_counts()
    print(
        f"\nCommunities with fewer than {min_rows} sales that lie entirely outside the overall "
        f"{trim.lower_quantile:.1%} to {trim.upper_quantile:.1%} range "
        f"({np.exp(lo):,.0f} to {np.exp(hi):,.0f} AED per sqm):"
    )
    for community, n in counts[counts < min_rows].items():
        part = log_pps[cleaned["community"] == community]
        if ((part < lo) | (part > hi)).all():
            print(f"  {community}: {n} sales, median {np.exp(part.median()):,.0f} AED per sqm")


def dubai_data_sample(path: Path) -> None:
    raw = schema.read_raw_csvs([path])
    print(f"data.dubai sample {path.name}: {len(raw):,} rows")
    print("Layout detected:", schema.detect_layout(raw.columns))
    years = pd.to_datetime(raw["instance_date"], errors="coerce").dt.year
    print("Rows per year (last five):", years.value_counts().sort_index().tail(5).to_dict())
    print(
        "Price-per-area columns dropped on read:", sorted(set(raw.columns) & schema.LEAKY_COLUMNS)
    )
    canonical = schema.to_canonical(raw)
    for col in ("transaction_date", "price_aed", "area_sqm", "community_en", "is_off_plan"):
        print(f"  {col}: {canonical[col].notna().mean():.1%} filled")
    cleaned, _ = clean.clean(canonical, config.DEFAULT_SETTINGS.cleaning)
    print(f"Apartment sales after the cleaning rules: {len(cleaned):,}")


def main(raw_dir: Path) -> None:
    raw = schema.read_raw_csvs(sorted(raw_dir.glob("transactions_*.csv")))
    df = schema.to_canonical(raw)
    print(f"Raw rows: {len(df):,}")
    names(df)
    procedures(df)
    shared_numbers(df)
    empty_columns(df)
    luxury(df)
    new_communities(df)
    shared_project_names(df)
    small_outlying_communities(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", nargs="?", type=Path, default=config.RAW_DIR)
    parser.add_argument(
        "--dubai-data-sample",
        nargs="?",
        type=Path,
        const=config.DATA_DIR / "raw" / "dubai_data" / "real_estate_transactions_sample.csv",
        help="check the data.dubai sample instead of the DLD export",
    )
    args = parser.parse_args()
    if args.dubai_data_sample:
        dubai_data_sample(args.dubai_data_sample)
    else:
        main(args.raw_dir)
