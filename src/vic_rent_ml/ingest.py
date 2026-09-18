"""Data ingestion for Homes Victoria rental tables and macroeconomic series."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

SHEET_TO_DWELLING = {
    "1 bedroom flat": ("flat", 1),
    "2 bedroom flat": ("flat", 2),
    "3 bedroom flat": ("flat", 3),
    "2 bedroom house": ("house", 2),
    "3 bedroom house": ("house", 3),
    "4 bedroom house": ("house", 4),
}
QUARTER_FROM_MONTH = {"Mar": 1, "Jun": 2, "Sep": 3, "Dec": 4}
GROUP_KEY = ["suburb_group", "apartment_type", "bedrooms"]


def parse_quarter_label(value) -> Optional[Tuple[int, int]]:
    text = str(value).strip()
    parts = text.split()
    if len(parts) != 2 or parts[0] not in QUARTER_FROM_MONTH:
        return None
    return int(parts[1]), QUARTER_FROM_MONTH[parts[0]]


def to_float(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "").replace("$", "")
    if text in {"", "-", "n.a.", "na", "NA", "."}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def melt_rent_sheet(
    path: Path, sheet: str, apartment_type: str, bedrooms: int
) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    labels = raw.iloc[1]
    body = raw.iloc[3:].copy()
    body[0] = body[0].ffill()
    body = body[body[1].notna()]
    body = body[~body[1].astype(str).str.contains("total", case=False, na=False)]

    frames = []
    for col in range(2, raw.shape[1], 2):
        parsed = parse_quarter_label(labels.iloc[col])
        if parsed is None:
            continue
        year, quarter = parsed
        chunk = pd.DataFrame(
            {
                "region": body[0].astype(str).str.strip(),
                "suburb_group": body[1].astype(str).str.strip(),
                "year": year,
                "quarter": quarter,
                "lease_count": body[col].map(to_float),
                "median": body[col + 1].map(to_float)
                if col + 1 < raw.shape[1]
                else np.nan,
            }
        )
        frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["apartment_type"] = apartment_type
    out["bedrooms"] = bedrooms
    return out.dropna(subset=["median"])


def attach_lags(rent: pd.DataFrame) -> pd.DataFrame:
    """Join exact calendar quarters; suppressed quarters must not shift time."""
    panel = rent.sort_values(GROUP_KEY + ["year", "quarter"]).copy()
    panel["period"] = panel["year"] * 4 + panel["quarter"]
    if panel.duplicated(GROUP_KEY + ["period"]).any():
        raise ValueError("Duplicate rent series/quarter")
    if not panel["quarter"].isin([1, 2, 3, 4]).all():
        raise ValueError("Invalid quarter")
    if not (panel["median"].dropna() > 0).all():
        raise ValueError("Non-positive rent")
    panel = panel.drop(columns=["median_lag_1", "median_lag_4"], errors="ignore")
    for lag in (1, 4):
        source = panel[GROUP_KEY + ["period", "median"]].copy()
        source["period"] += lag
        panel = panel.merge(
            source.rename(columns={"median": f"median_lag_{lag}"}),
            on=GROUP_KEY + ["period"], how="left", validate="one_to_one",
        )
    panel["delta"] = panel["median"] - panel["median_lag_1"]
    panel["year_frac"] = panel["year"] + (panel["quarter"] - 1) / 4.0
    return panel.dropna(
        subset=["median", "median_lag_1", "median_lag_4"]
    ).copy()


def load_rent_tables(path: Path) -> pd.DataFrame:
    with pd.ExcelFile(path) as workbook:
        missing = set(SHEET_TO_DWELLING) - set(workbook.sheet_names)
    if missing:
        raise ValueError(f"Missing dwelling sheets: {sorted(missing)}")
    rent = pd.concat([
        melt_rent_sheet(path, sheet, kind, beds)
        for sheet, (kind, beds) in SHEET_TO_DWELLING.items()
    ], ignore_index=True)
    rent["period"] = rent["year"] * 4 + rent["quarter"]
    return rent


def build_panel_from_workbook(path: Path, postcode_map: dict) -> pd.DataFrame:
    rent = load_rent_tables(path)
    rent["postcodes"] = rent["suburb_group"].map(postcode_map).fillna("missing")
    return attach_lags(rent)


def build_panel_from_transformed(path: Path, region_map: dict | None = None) -> pd.DataFrame:
    """Import the application's versioned historical CSV, excluding added covariates."""
    source = pd.read_csv(path)
    panel = source.rename(columns={"suburb": "suburb_group", "count": "lease_count"}).copy()
    dates = panel["date"].map(parse_quarter_label)
    if dates.isna().any():
        raise ValueError("Unrecognised quarter labels in historical CSV")
    panel[["year", "quarter"]] = pd.DataFrame(dates.tolist(), index=panel.index)
    panel["median"] = panel["median"].map(to_float)
    panel["region"] = panel["postcodes"].map(
        lambda pc: (region_map or {}).get(str(int(pc)), "Unknown") if pd.notna(pc) else "Unknown"
    )
    columns = GROUP_KEY + ["year", "quarter", "median", "postcodes", "lease_count", "region"]
    panel = panel[columns].dropna(subset=["median"])
    panel = panel[~panel["suburb_group"].str.contains("total", case=False)]
    return attach_lags(panel)
