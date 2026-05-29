"""
transform_clio_to_2025.py
=========================
Converts OGC Clio Activities XLSX exports into the same CSV schema used by
the 2024 LeanLaw data files.

Outputs (written to /2025 and /2026 folders):
  - ATTORNEY_CLIENTS_<year>.csv  — Sum of billing Amount by Client x Attorney x Month
  - PIVOT_SOURCE_1_<year>.csv    — Sum of Hours by Attorney x Month (all clients combined)

Usage:
  python transform_clio_to_2025.py

Notes on hours calculation:
  Clio only exports billing amounts (Total column), not raw hours.
  For "Hourly time entry" rows, hours are back-calculated as: Total / hourly_rate.
  Rates are sourced first from RATE_OVERRIDES (manual), then from 2024 median rates,
  then from DEFAULT_RATE. "Flat rate time entry" rows are excluded from hours
  (they cannot be reliably converted to time).
  Hard cost / Soft cost / Expense rows are excluded entirely.
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION — edit these if rates or file paths change
# =============================================================================

# Absolute base path (required because OneDrive folders may be read-only for
# relative-path writes but can be reached via PowerShell Copy-Item)
_BASE = os.path.dirname(os.path.abspath(__file__))

CLIO_2025_FILE   = os.path.join(_BASE, "OGC Clio Activities 2025.xlsx")
CLIO_2026_FILE   = os.path.join(_BASE, "OGC Clio Activities 1.1 to 5.29.26.xlsx")
ATTORNEY_META    = os.path.join(_BASE, "2024", "ATTORNEY_PG_AND_HRS_2024.csv")
RAW_2024_FILE    = os.path.join(_BASE, "2024", "SIX_FULL_MOS_2024.csv")

# Output directories.
# The project lives on OneDrive Files-On-Demand, which blocks Python from
# creating NEW files directly in cloud-only folders.  We therefore write to a
# local path outside OneDrive and print copy instructions at the end.
# If you have made the project folder "Always available locally" in OneDrive,
# you can change these back to:
#   OUTPUT_2025_DIR = os.path.join(_BASE, "2025")
#   OUTPUT_2026_DIR = os.path.join(_BASE, "2026")
_LOCAL_OUTPUT = os.path.join(os.path.expanduser("~"), "AppData", "Local", "OGC_Output")
OUTPUT_2025_DIR  = os.path.join(_LOCAL_OUTPUT, "2025")
OUTPUT_2026_DIR  = os.path.join(_LOCAL_OUTPUT, "2026")

# Manual rate overrides (attorney name as it appears in Clio → $/hour).
# These take priority over everything else.
RATE_OVERRIDES: dict[str, float] = {
    # Example: "Jordan Karp": 495.0,
}

DEFAULT_RATE = 425.0   # fallback for attorneys with no rate data

# Clio sometimes uses different names than LeanLaw / the metadata file.
# Map  Clio name  →  canonical 2024 name  so metadata (PG, STATE, TARGET) is found.
ATTORNEY_NAME_MAP: dict[str, str] = {
    "Eddie Litton":  "W. Edwin Litton",
    "Mich\u00e8le Linde": "Mich\u00e8le Linde",   # encoding safety
}

# =============================================================================
# CONSTANTS
# =============================================================================

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_NUM = {m: i+1 for i, m in enumerate(MONTHS)}
NUM_TO_MONTH = {i+1: m for i, m in enumerate(MONTHS)}

# Entry types that represent actual attorney time
TIME_ENTRY_TYPES = {"Hourly time entry", "Flat rate time entry"}
HOURLY_TYPE      = "Hourly time entry"

# =============================================================================
# DATA LOADING
# =============================================================================

def load_clio(filepath: str) -> pd.DataFrame:
    """Read a Clio Activities XLSX and return normalised time-entry rows."""
    df = pd.read_excel(filepath, sheet_name="Report")
    df["Date"]      = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"]      = df["Date"].dt.year
    df["Month_Num"] = df["Date"].dt.month
    df["Month"]     = df["Month_Num"].map(NUM_TO_MONTH)
    df["User"]      = df["User"].str.strip()
    # Apply name mapping
    df["User"] = df["User"].replace(ATTORNEY_NAME_MAP)
    # Keep only time-based entries
    df = df[df["Type"].isin(TIME_ENTRY_TYPES)].copy()
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce").fillna(0)
    return df


def load_attorney_rates() -> dict[str, float]:
    """Return {attorney_name: median_hourly_rate} from 2024 raw data."""
    try:
        df = pd.read_csv(RAW_2024_FILE, low_memory=False)
        df["Rate"]  = pd.to_numeric(df["Rate"],  errors="coerce")
        time_rows   = df[(df["Activity Type"] == "Time") & (df["Rate"] > 0)]
        rates       = time_rows.groupby("Associated Attorney")["Rate"].median()
        return rates.to_dict()
    except Exception as exc:
        print(f"  [warn] Could not load 2024 rates: {exc}. Using DEFAULT_RATE for all.")
        return {}


def load_attorney_metadata() -> dict[str, dict]:
    """Return {attorney_name: {TARGET, PG, STATE}} from 2024 metadata file."""
    try:
        meta = pd.read_csv(ATTORNEY_META)
        result = {}
        for _, row in meta.iterrows():
            name = str(row.get("Attorney Name", "")).strip()
            if name:
                result[name] = {
                    "TARGET": row.get("🎚️ Target Hours / Month", ""),
                    "PG":     row.get("Practice Area (Primary)", ""),
                    "STATE":  row.get("Mailing State (Abbrev)", ""),
                }
        return result
    except Exception as exc:
        print(f"  [warn] Could not load attorney metadata: {exc}")
        return {}

# =============================================================================
# HOURS CALCULATION
# =============================================================================

def build_rate_table(attorneys: list[str],
                     rates_2024: dict[str, float]) -> dict[str, float]:
    """
    Build a per-attorney rate lookup, merging:
      1. RATE_OVERRIDES  (highest priority)
      2. 2024 median rates
      3. DEFAULT_RATE    (fallback)
    """
    table = {}
    for atty in attorneys:
        if atty in RATE_OVERRIDES:
            table[atty] = RATE_OVERRIDES[atty]
        elif atty in rates_2024:
            table[atty] = rates_2024[atty]
        else:
            # Check if canonical 2024 name is available
            canonical = ATTORNEY_NAME_MAP.get(atty, atty)
            if canonical in rates_2024:
                table[atty] = rates_2024[canonical]
            else:
                table[atty] = DEFAULT_RATE
    return table


def calc_hours(df: pd.DataFrame, rate_table: dict[str, float]) -> pd.Series:
    """
    For hourly entries: hours = Total / rate.
    For flat-rate entries: not convertible to hours → returns 0.
    """
    rates  = df["User"].map(rate_table).fillna(DEFAULT_RATE)
    hourly = df["Type"] == HOURLY_TYPE
    hours  = pd.Series(0.0, index=df.index)
    # Guard against divide-by-zero
    safe_rates = rates.where(rates > 0, DEFAULT_RATE)
    hours[hourly] = df.loc[hourly, "Total"] / safe_rates[hourly]
    return hours.round(4)

# =============================================================================
# PIVOT BUILDERS
# =============================================================================

def pivot_attorney_clients(df: pd.DataFrame,
                           meta: dict[str, dict]) -> pd.DataFrame:
    """
    ATTORNEY_CLIENTS schema:
      Client Name | Associated Attorney | TARGET | PG | STATE | Jan…Dec | Grand Total
    Values: Sum of billing Amount (Total column).
    """
    grouped = (df.groupby(["Client", "User", "Month"], observed=True)["Total"]
                 .sum()
                 .reset_index())
    grouped.columns = ["Client Name", "Associated Attorney", "Month", "Amount"]

    pivot = grouped.pivot_table(
        index=["Client Name", "Associated Attorney"],
        columns="Month",
        values="Amount",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot.columns.name = None

    # Ensure all 12 months are present
    for m in MONTHS:
        if m not in pivot.columns:
            pivot[m] = 0.0

    pivot["Grand Total"] = pivot[MONTHS].sum(axis=1)

    # Attach metadata
    pivot["TARGET"] = pivot["Associated Attorney"].map(
        lambda x: meta.get(x, {}).get("TARGET", ""))
    pivot["PG"]     = pivot["Associated Attorney"].map(
        lambda x: meta.get(x, {}).get("PG", ""))
    pivot["STATE"]  = pivot["Associated Attorney"].map(
        lambda x: meta.get(x, {}).get("STATE", ""))

    col_order = (["Client Name", "Associated Attorney", "TARGET", "PG", "STATE"]
                 + MONTHS + ["Grand Total"])
    pivot = pivot[col_order].sort_values(["Client Name", "Associated Attorney"])

    # Remove rows where all months are zero (e.g. pure flat-rate clients if excluded)
    pivot = pivot[pivot["Grand Total"] > 0].reset_index(drop=True)
    return pivot


def pivot_source_1(df: pd.DataFrame,
                   meta: dict[str, dict],
                   rate_table: dict[str, float]) -> pd.DataFrame:
    """
    PIVOT_SOURCE_1 schema:
      Associated Attorney | TARGET | PG | STATE | Jan…Dec | Grand Total
    Values: Sum of estimated Hours.
    Only hourly entries contribute (flat-rate excluded — see module docstring).
    """
    df = df.copy()
    df["Hours"] = calc_hours(df, rate_table)

    grouped = (df[df["Hours"] > 0]
               .groupby(["User", "Month"], observed=True)["Hours"]
               .sum()
               .reset_index())
    grouped.columns = ["Associated Attorney", "Month", "Hours"]

    pivot = grouped.pivot_table(
        index="Associated Attorney",
        columns="Month",
        values="Hours",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot.columns.name = None

    for m in MONTHS:
        if m not in pivot.columns:
            pivot[m] = 0.0

    pivot["Grand Total"] = pivot[MONTHS].sum(axis=1).round(2)

    pivot["TARGET"] = pivot["Associated Attorney"].map(
        lambda x: meta.get(x, {}).get("TARGET", ""))
    pivot["PG"]     = pivot["Associated Attorney"].map(
        lambda x: meta.get(x, {}).get("PG", ""))
    pivot["STATE"]  = pivot["Associated Attorney"].map(
        lambda x: meta.get(x, {}).get("STATE", ""))

    col_order = (["Associated Attorney", "TARGET", "PG", "STATE"]
                 + MONTHS + ["Grand Total"])
    pivot = pivot[col_order].sort_values("Associated Attorney")
    return pivot.reset_index(drop=True)

# =============================================================================
# CSV WRITERS — replicating the exact 2024 pivot-table export format
# =============================================================================

def _make_empty_row(n_cols: int) -> list:
    return [""] * n_cols


def _write_csv(df_out: pd.DataFrame, filepath: str) -> None:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_out.to_csv(filepath, index=False, encoding="utf-8-sig")


def save_attorney_clients(pivot: pd.DataFrame, filepath: str) -> None:
    """
    Write ATTORNEY_CLIENTS in the 2024 Excel-pivot-export format.

    Row layout (18 columns total, same as 2024):
      Rows 1-14  : blank
      Row 15     : "Sum of Amount" | … | "Months (Service Date)" | …
      Row 16     : blank×5 | Jan | Feb | … | Dec | Grand Total
      Row 17     : blank
      Row 18     : Client Name | Associated Attorney | TARGET | PG | STATE | …
      Row 19+    : data
    """
    N = 18   # number of columns

    rows = []
    # Rows 1-14 blank (first row becomes the CSV header — all unnamed cols)
    for _ in range(14):
        rows.append(_make_empty_row(N))

    # Row 15: sum label + months label
    r15 = _make_empty_row(N)
    r15[0] = "Sum of Amount"
    r15[5] = "Months (Service Date)"
    r15[6] = "Days (Service Date)"
    r15[7] = "Service Date"
    rows.append(r15)

    # Row 16: month names (cols 5-17)
    r16 = _make_empty_row(N)
    for i, m in enumerate(MONTHS):
        r16[5 + i] = m
    r16[17] = "Grand Total"
    rows.append(r16)

    # Row 17: blank
    rows.append(_make_empty_row(N))

    # Row 18: column labels
    r18 = _make_empty_row(N)
    r18[0] = "Client Name"
    r18[1] = "Associated Attorney"
    r18[2] = "TARGET"
    r18[3] = "PG"
    r18[4] = "STATE"
    rows.append(r18)

    # Data rows
    for _, row in pivot.iterrows():
        r = _make_empty_row(N)
        r[0] = row["Client Name"]
        r[1] = row["Associated Attorney"]
        r[2] = row["TARGET"]
        r[3] = row["PG"]
        r[4] = row["STATE"]
        for i, m in enumerate(MONTHS):
            val = row.get(m, 0)
            r[5 + i] = val if val != 0 else ""
        r[17] = row["Grand Total"]
        rows.append(r)

    header = [f"Unnamed: {i}" for i in range(N)]
    df_out = pd.DataFrame(rows, columns=header)
    _write_csv(df_out, filepath)
    print(f"  Saved {len(pivot):,} rows -> {filepath}")


def save_pivot_source_1(pivot: pd.DataFrame, filepath: str) -> None:
    """
    Write PIVOT_SOURCE_1 in the 2024 Excel-pivot-export format.

    Row layout (19 columns total, same as 2024):
      Rows 1-14  : blank
      Row 15     : "Sum of Hours" | … | "Months (Service Date)" | …
      Row 16     : Associated Attorney | TARGET | PG | STATE | Jan…Dec | Grand Total | …
      Row 17+    : data
    """
    N = 19

    rows = []
    for _ in range(14):
        rows.append(_make_empty_row(N))

    r15 = _make_empty_row(N)
    r15[0] = "Sum of Hours"
    r15[4] = "Months (Service Date)"
    rows.append(r15)

    # Row 16 is the data header (same row as first data descriptor in 2024)
    r16 = _make_empty_row(N)
    r16[0]  = "Associated Attorney"
    r16[1]  = "TARGET"
    r16[2]  = "PG"
    r16[3]  = "STATE"
    for i, m in enumerate(MONTHS):
        r16[4 + i] = m
    r16[16] = "Grand Total"
    rows.append(r16)

    # Data rows
    for _, row in pivot.iterrows():
        r = _make_empty_row(N)
        r[0] = row["Associated Attorney"]
        r[1] = row["TARGET"]
        r[2] = row["PG"]
        r[3] = row["STATE"]
        for i, m in enumerate(MONTHS):
            val = row.get(m, 0)
            r[4 + i] = round(val, 4) if val != 0 else ""
        r[16] = row["Grand Total"]
        r[17] = row["Associated Attorney"]   # repeated col as in 2024
        r[18] = 1.0
        rows.append(r)

    header = [f"Unnamed: {i}" for i in range(N)]
    df_out = pd.DataFrame(rows, columns=header)
    _write_csv(df_out, filepath)
    print(f"  Saved {len(pivot):,} rows -> {filepath}")

# =============================================================================
# MAIN
# =============================================================================

def process_year(clio_file: str,
                 output_dir: str,
                 year_label: str,
                 rates_2024: dict[str, float],
                 meta: dict[str, dict]) -> None:
    print(f"\n{'='*60}")
    print(f"Processing: {clio_file}  ->  /{output_dir}/")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    df = load_clio(clio_file)
    print(f"  Loaded {len(df):,} time entries "
          f"({df['Year'].min()} – {df['Year'].max()})")

    attorneys = df["User"].unique().tolist()
    rate_table = build_rate_table(attorneys, rates_2024)

    # Report attorneys using the fallback rate
    fallback = [a for a in attorneys
                if a not in RATE_OVERRIDES and a not in rates_2024
                and ATTORNEY_NAME_MAP.get(a, a) not in rates_2024]
    if fallback:
        print(f"\n  [info] {len(fallback)} attorney(s) not found in 2024 rate data "
              f"- using DEFAULT_RATE (${DEFAULT_RATE}/hr):")
        for a in sorted(fallback):
            print(f"         {a}")

    # ATTORNEY_CLIENTS
    print("\n  Building ATTORNEY_CLIENTS ...")
    ac = pivot_attorney_clients(df, meta)
    save_attorney_clients(
        ac, os.path.join(output_dir, f"ATTORNEY_CLIENTS_{year_label}.csv"))

    # PIVOT_SOURCE_1
    print("  Building PIVOT_SOURCE_1 ...")
    p1 = pivot_source_1(df, meta, rate_table)
    save_pivot_source_1(
        p1, os.path.join(output_dir, f"PIVOT_SOURCE_1_{year_label}.csv"))

    # Quick sanity summary
    print(f"\n  Summary for {year_label}:")
    print(f"    Unique clients:   {df['Client'].nunique():,}")
    print(f"    Unique attorneys: {df['User'].nunique():,}")
    total_amount = df["Total"].sum()
    print(f"    Total billed:    ${total_amount:,.2f}")
    hourly_df = df[df["Type"] == HOURLY_TYPE].copy()
    hourly_df["Hours"] = calc_hours(hourly_df, rate_table)
    total_hours = hourly_df["Hours"].sum()
    print(f"    Est. total hours: {total_hours:,.1f}  "
          f"(hourly entries only; flat-rate excluded)")


def main() -> None:
    print("Loading shared reference data ...")
    rates_2024 = load_attorney_rates()
    meta       = load_attorney_metadata()
    print(f"  Rate table loaded: {len(rates_2024)} attorneys from 2024 data")
    print(f"  Metadata loaded:   {len(meta)} attorneys")

    process_year(CLIO_2025_FILE, OUTPUT_2025_DIR, "2025", rates_2024, meta)
    process_year(CLIO_2026_FILE, OUTPUT_2026_DIR, "2026", rates_2024, meta)

    print("\n" + "="*60)
    print("OUTPUT LOCATION (local, outside OneDrive):")
    print(f"  {_LOCAL_OUTPUT}")
    print()
    print("To move files into the project, run this in PowerShell:")
    print(f'  Copy-Item "{OUTPUT_2025_DIR}" -Destination "{os.path.join(_BASE, "2025")}" -Recurse -Force')
    print(f'  Copy-Item "{OUTPUT_2026_DIR}" -Destination "{os.path.join(_BASE, "2026")}" -Recurse -Force')
    print("="*60)


if __name__ == "__main__":
    main()
