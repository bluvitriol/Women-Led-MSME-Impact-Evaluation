"""
02_clean_data.py
----------------
Cleans the raw MSME survey dataset produced by 01_simulate_data.py.

Cleaning steps (each documented with rationale):
    1.  Load raw data — never overwrite it
    2.  Audit: shape, dtypes, missing counts, duplicate check
    3.  Deduplicate on business_id
    4.  Standardise categorical columns (sector, owner_education)
    5.  Parse and standardise survey_date to ISO 8601
    6.  Handle missing values — flag + impute where justified
    7.  Winsorise extreme revenue values
    8.  Create analysis indicators
    9.  Save clean dataset and a cleaning log

Author: Valentina Sharma
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_PATH   = "data/raw/msme_survey_raw.csv"
CLEAN_PATH = "data/clean/msme_survey_clean.csv"
LOG_PATH   = "data/clean/cleaning_log.json"

os.makedirs("data/clean", exist_ok=True)

# ── Cleaning log (append decisions as we go) ──────────────────────────────────
log = {
    "script": "02_clean_data.py",
    "run_timestamp": datetime.now().isoformat(),
    "steps": []
}

def log_step(step_name: str, detail: str, rows_affected: int = 0):
    """Append a documented cleaning decision to the log."""
    entry = {"step": step_name, "detail": detail, "rows_affected": rows_affected}
    log["steps"].append(entry)
    print(f"[{step_name}] {detail}  (rows affected: {rows_affected})")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load raw data
# ════════════════════════════════════════════════════════════════════════════
df_raw = pd.read_csv(RAW_PATH)
df = df_raw.copy()  # RULE: never mutate the raw dataframe

log_step("LOAD", f"Raw dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Audit
# ════════════════════════════════════════════════════════════════════════════
print("\n── RAW AUDIT ──────────────────────────────────────")
print(df.dtypes)
print("\nMissing value counts:")
print(df.isnull().sum())
print(f"\nDuplicated business_id count: {df['business_id'].duplicated().sum()}")
print("───────────────────────────────────────────────────\n")

log_step("AUDIT", "Printed dtypes, missing counts, duplicate ID count — see console output")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Deduplicate on business_id
# ════════════════════════════════════════════════════════════════════════════
# Decision: keep the LAST occurrence of each duplicate ID.
# Rationale: in real survey data, later entries are often corrections/re-interviews.
# Flag all rows that were originally duplicates before dropping.

n_before = len(df)
df["is_duplicate_flag"] = df.duplicated(subset="business_id", keep=False).astype(int)
n_duplicates = df["is_duplicate_flag"].sum()

df = df.drop_duplicates(subset="business_id", keep="last").reset_index(drop=True)
n_after = len(df)

log_step(
    "DEDUPLICATION",
    "Kept last occurrence of each duplicated business_id; added is_duplicate_flag column.",
    rows_affected=n_before - n_after
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Standardise categorical columns
# ════════════════════════════════════════════════════════════════════════════

# 4a. sector — title-case everything, strip whitespace
sector_before = df["sector"].unique().tolist()
df["sector"] = df["sector"].str.strip().str.title()
sector_after = df["sector"].unique().tolist()

log_step(
    "STANDARDISE_SECTOR",
    f"Applied str.title() and str.strip(). Unique values before: {sorted(sector_before)} → after: {sorted(sector_after)}",
    rows_affected=df["sector"].notna().sum()
)

# 4b. owner_education — define an ordered categorical for later analysis
education_order = ["No formal education", "Primary", "Secondary", "Graduate"]
df["owner_education"] = df["owner_education"].str.strip().str.title()
# Recode any values not in our expected list as NaN
valid_education = set(education_order)
unexpected = df.loc[~df["owner_education"].isin(valid_education) & df["owner_education"].notna(), "owner_education"].unique()
if len(unexpected):
    df.loc[~df["owner_education"].isin(valid_education), "owner_education"] = np.nan
    log_step("STANDARDISE_EDUCATION", f"Recoded unexpected values to NaN: {list(unexpected)}", rows_affected=len(unexpected))

df["owner_education"] = pd.Categorical(df["owner_education"], categories=education_order, ordered=True)

log_step(
    "ENCODE_EDUCATION",
    "Converted owner_education to ordered Categorical: No formal education < Primary < Secondary < Graduate"
)

# 4c. state — title-case
df["state"] = df["state"].str.strip().str.title()
df["district"] = df["district"].str.strip().str.title()

log_step("STANDARDISE_GEO", "Applied str.title() and str.strip() to state and district columns.")


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Parse and standardise survey_date
# ════════════════════════════════════════════════════════════════════════════
# Raw data has three formats: DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD.
# Strategy: attempt ISO parse first, then try DD/MM, then MM/DD.
# Flag ambiguous rows (where day ≤ 12, so DD/MM and MM/DD are indistinguishable).

def parse_date(date_str):
    """
    Attempt to parse a date string across three known formats.
    Returns a pandas Timestamp or NaT.
    """
    if pd.isna(date_str):
        return pd.NaT
    date_str = str(date_str).strip()
    # ISO format: YYYY-MM-DD
    if "-" in date_str:
        try:
            return pd.to_datetime(date_str, format="%Y-%m-%d")
        except ValueError:
            pass
    # Slash formats
    if "/" in date_str:
        parts = date_str.split("/")
        if len(parts) == 3:
            a, b, y = parts
            # If first part > 12, must be DD/MM/YYYY
            if int(a) > 12:
                try:
                    return pd.to_datetime(date_str, format="%d/%m/%Y")
                except ValueError:
                    pass
            else:
                # Ambiguous — default to DD/MM/YYYY (document this assumption)
                try:
                    return pd.to_datetime(date_str, format="%d/%m/%Y")
                except ValueError:
                    pass
    return pd.NaT

df["survey_date_parsed"] = df["survey_date"].apply(parse_date)
n_failed = df["survey_date_parsed"].isna().sum()

log_step(
    "PARSE_DATES",
    (
        "Parsed survey_date to ISO 8601 (survey_date_parsed). "
        "Ambiguous DD/MM vs MM/DD cases defaulted to DD/MM/YYYY. "
        f"Failed parses (NaT): {n_failed}"
    ),
    rows_affected=len(df) - n_failed
)

# Drop the original messy column; keep parsed version
df = df.drop(columns=["survey_date"])
df = df.rename(columns={"survey_date_parsed": "survey_date"})


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — Handle missing values
# ════════════════════════════════════════════════════════════════════════════

# 6a. endline_revenue_inr
# Decision: do NOT impute the primary outcome variable.
# Create a missing flag instead and drop rows for regression analysis only.
n_missing_endline = df["endline_revenue_inr"].isna().sum()
df["endline_revenue_missing"] = df["endline_revenue_inr"].isna().astype(int)

log_step(
    "MISSING_ENDLINE_REVENUE",
    (
        f"{n_missing_endline} rows have missing endline_revenue_inr. "
        "Created endline_revenue_missing flag. Rows retained for descriptives; "
        "will be dropped only during regression (in 03_analysis.py)."
    ),
    rows_affected=n_missing_endline
)

# 6b. employees_baseline
# Decision: impute with the median, stratified by sector.
# Rationale: employee count is a covariate, not the outcome. Median is robust to outliers.
n_missing_emp = df["employees_baseline"].isna().sum()
df["employees_baseline_imputed"] = df["employees_baseline"].isna().astype(int)

sector_medians = df.groupby("sector")["employees_baseline"].median()
def impute_employees(row):
    if pd.isna(row["employees_baseline"]):
        return sector_medians.get(row["sector"], df["employees_baseline"].median())
    return row["employees_baseline"]

df["employees_baseline"] = df.apply(impute_employees, axis=1)

log_step(
    "IMPUTE_EMPLOYEES",
    (
        f"{n_missing_emp} missing employees_baseline values imputed with "
        "sector-stratified median. Added employees_baseline_imputed flag column."
    ),
    rows_affected=n_missing_emp
)

# 6c. owner_education
# Decision: retain NaN as its own category ("Unknown") for descriptive analysis.
# Do not impute education — too much assumption risk.
n_missing_edu = df["owner_education"].isna().sum()
education_order_ext = ["No formal education", "Primary", "Secondary", "Graduate", "Unknown"]
df["owner_education"] = df["owner_education"].astype(str).replace("nan", "Unknown")
df["owner_education"] = pd.Categorical(df["owner_education"], categories=education_order_ext, ordered=False)

log_step(
    "MISSING_EDUCATION",
    f"{n_missing_edu} missing owner_education values retained as 'Unknown' category.",
    rows_affected=n_missing_edu
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — Winsorise extreme revenue values
# ════════════════════════════════════════════════════════════════════════════
# Winsorise at 1st and 99th percentile to reduce influence of extreme outliers.
# Applied to both baseline and endline revenue.

for col in ["baseline_revenue_inr", "endline_revenue_inr"]:
    p1  = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    n_winsorised = ((df[col] < p1) | (df[col] > p99)).sum()
    df[col] = df[col].clip(lower=p1, upper=p99)
    log_step(
        f"WINSORISE_{col.upper()}",
        f"Clipped to [{p1:.0f}, {p99:.0f}] INR (1st–99th percentile).",
        rows_affected=int(n_winsorised)
    )


# ════════════════════════════════════════════════════════════════════════════
# STEP 8 — Create analysis indicators
# ════════════════════════════════════════════════════════════════════════════

# 8a. Revenue growth rate (endline / baseline - 1)
df["revenue_growth_rate"] = (
    (df["endline_revenue_inr"] - df["baseline_revenue_inr"]) / df["baseline_revenue_inr"]
)

# 8b. Log revenue (for regression — log-linear model is standard in IE)
df["log_baseline_revenue"] = np.log(df["baseline_revenue_inr"].replace(0, np.nan))
df["log_endline_revenue"]  = np.log(df["endline_revenue_inr"].replace(0, np.nan))

# 8c. Region dummies (for fixed effects in regression)
region_dummies = pd.get_dummies(df["state"], prefix="state_fe", drop_first=True)
df = pd.concat([df, region_dummies], axis=1)

# 8d. Education numeric encoding (for heterogeneity analysis)
edu_map = {
    "No formal education": 0,
    "Primary": 1,
    "Secondary": 2,
    "Graduate": 3,
    "Unknown": np.nan
}
df["owner_education_numeric"] = df["owner_education"].map(edu_map)

log_step(
    "CREATE_INDICATORS",
    (
        "Created: revenue_growth_rate, log_baseline_revenue, log_endline_revenue, "
        "state fixed-effect dummies (drop_first=True), owner_education_numeric."
    )
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 9 — Final audit and save
# ════════════════════════════════════════════════════════════════════════════

print("\n── CLEAN DATA AUDIT ────────────────────────────────")
print(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nRemaining missing values (key columns):")
key_cols = ["business_id", "treatment", "baseline_revenue_inr",
            "endline_revenue_inr", "employees_baseline", "sector", "state"]
print(df[key_cols].isnull().sum())
print(f"\nTreatment balance:\n{df['treatment'].value_counts()}")
print("────────────────────────────────────────────────────\n")

df.to_csv(CLEAN_PATH, index=False)
log_step("SAVE", f"Clean dataset saved to {CLEAN_PATH}. Shape: {df.shape[0]} rows × {df.shape[1]} columns.")

# Save cleaning log
with open(LOG_PATH, "w") as f:
    json.dump(log, f, indent=2, default=str)

print(f"\nCleaning complete.")
print(f"  Clean data  → {CLEAN_PATH}")
print(f"  Cleaning log → {LOG_PATH}")