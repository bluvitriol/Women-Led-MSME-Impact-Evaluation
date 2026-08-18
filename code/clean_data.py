import pandas as pd
import numpy as np
import os

# Loading raw data

raw_path = "data/raw/msme_survey_raw.csv"
df = pd.read_csv(raw_path)

print("RAW DATA SUMMARY")
print(f"Rows: {len(df)}")
print(f"Columns: {df.shape[1]}")
print(f"\nMissing values per column:")
print(df.isnull().sum())
print(f"\nData types:")
print(df.dtypes)

# Keep a copy to compare later
original_row_count = len(df)


# STEP 1 - Remove duplicate business IDs
# Keeping the first occurrence and dropping the rest.

duplicates_found = df.duplicated(subset="business_id").sum()
print(f"\n[Step 1] Duplicates found: {duplicates_found}")

df = df.drop_duplicates(subset="business_id", keep="first")
print(f"[Step 1] Rows after removing duplicates: {len(df)}")


# STEP 2 - Standardize the 'sector' column
# inconsistent "Retail", "RETAIL", "retail"
# strip whitespace and convert to Title Case

print(f"\n[Step 2] Unique sector values BEFORE cleaning:")
print(df["sector"].unique())

df["sector"] = df["sector"].str.strip().str.title()

print("[Step 2] Unique sector values AFTER cleaning:")
print(df["sector"].unique())


# STEP 3 - Standardize the 'survey_date' column
# dates are in 3 formats - DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD
# parsing with dayfirst=True (most common in Indian survey data),
#      then convert to a standard ISO format (YYYY-MM-DD)
# Note: flagging rows where parsing failed so they can be reviewed manually

print(f"\n[Step 3] Sample raw survey dates:")
print(df["survey_date"].head(10).tolist())

df["survey_date_clean"] = pd.to_datetime(
    df["survey_date"],
    dayfirst=True,        # assumes DD/MM/YYYY when ambiguous
    errors="coerce"       # turns unparseable dates into NaT instead of crashing
)

failed_dates = df["survey_date_clean"].isna().sum()
print(f"[Step 3] Dates that could not be parsed: {failed_dates}")
# Drop the original messy column; keep only the clean one
df = df.drop(columns=["survey_date"])
df = df.rename(columns={"survey_date_clean": "survey_date"})


# STEP 4 - Handle missing values
# handling each column differently based on what makes sense for the data.
# All decisions are documented so the cleaning is reproducible.

print(f"\n[Step 4] Handling missing values")

# 4a. endline_revenue_inr - this is our main outcome variable.
#     NOT imputing it (that would bias results).
#     flagging these rows with a separate indicator column instead.
df["endline_revenue_missing"] = df["endline_revenue_inr"].isna().astype(int)
missing_endline = df["endline_revenue_missing"].sum()
print(f"  - endline_revenue: {missing_endline} missing → flagged with indicator column")

# 4b. employees_baseline - we impute with the median (robust to outliers)
median_employees = df["employees_baseline"].median()
df["employees_baseline"] = df["employees_baseline"].fillna(median_employees)
print(f"  - employees_baseline: imputed with median ({median_employees})")

# 4c. owner_education - we fill with "Unknown" (a legitimate category)
df["owner_education"] = df["owner_education"].fillna("Unknown")
print(f"  - owner_education: missing filled with 'Unknown'")


# STEP 5 - Create analysis-ready indicator variables

print(f"\n[Step 5] Creating indicator variables...")

# Revenue growth rate (only calculable where endline is not missing)
df["revenue_growth_rate"] = (
    (df["endline_revenue_inr"] - df["baseline_revenue_inr"])
    / df["baseline_revenue_inr"]
)
print(f"  - revenue_growth_rate: created (% change from baseline to endline)")

# Log of baseline revenue (log-linearizes skewed revenue data - standard in IE)
df["log_baseline_revenue"] = np.log(df["baseline_revenue_inr"])
print(f"  - log_baseline_revenue: created (log of baseline revenue)")

# Log of endline revenue
df["log_endline_revenue"] = np.log(df["endline_revenue_inr"])
print(f"  - log_endline_revenue: created (log of endline revenue)")

# Dummy variables for owner education (needed for regression)
education_dummies = pd.get_dummies(
    df["owner_education"],
    prefix="edu",
    drop_first=True    # drop first category to avoid multicollinearity
)
df = pd.concat([df, education_dummies], axis=1)
print(f"  - education dummies created: {list(education_dummies.columns)}")

# Dummy variables for state (region fixed effects)
state_dummies = pd.get_dummies(
    df["state"],
    prefix="state",
    drop_first=True
)
df = pd.concat([df, state_dummies], axis=1)
print(f"  - state dummies created: {list(state_dummies.columns)}")


# STEP 6 - Final checks before saving

print(f"\n[Step 6] Final data checks...")
print(f"  - Rows in raw data:   {original_row_count}")
print(f"  - Rows in clean data: {len(df)}")
print(f"  - Rows dropped:       {original_row_count - len(df)}")
print(f"  - Columns:            {df.shape[1]}")
print(f"\nFinal missing value check:")
print(df.isnull().sum()[df.isnull().sum() > 0])


# STEP 7 - Save clean data
os.makedirs("../data/clean", exist_ok=True)
clean_path = "../data/clean/msme_survey_clean2.csv"
df.to_csv(clean_path, index=False)

print(f"\n[Step 7] Clean dataset saved to: {clean_path}")
print("CLEANING COMPLETE")
