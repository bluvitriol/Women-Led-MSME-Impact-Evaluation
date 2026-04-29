import numpy as np
import pandas as pd

np.random.seed(42)  # for reproducibility

N = 520  # oversampling slightly

# --- Basic identifiers ---
business_ids = [f"MSME{str(i).zfill(4)}" for i in range(1, N + 1)]

# Introducing ~20 duplicate IDs to simulate messy raw data
duplicate_indices = np.random.choice(range(N), size=20, replace=False)
for idx in duplicate_indices:
    business_ids[idx] = business_ids[idx - 1]  # copy previous ID

# --- Geographic variables ---
states = ["Rajasthan", "Uttar Pradesh", "Maharashtra", "Bihar", "Gujarat"]
districts = {
    "Rajasthan": ["Jaipur", "Jodhpur", "Ajmer"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra"],
    "Maharashtra": ["Pune", "Nagpur", "Nashik"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur"],
    "Gujarat": ["Surat", "Vadodara", "Rajkot"]
}

state_col = np.random.choice(states, size=N)
district_col = [np.random.choice(districts[s]) for s in state_col]

# --- Treatment assignment (randomized at district level) ---
# 60% treatment, 40% control — simulating imperfect compliance
treatment = np.random.choice([1, 0], size=N, p=[0.6, 0.4])

# --- Baseline characteristics ---
baseline_revenue = np.random.lognormal(mean=10.5, sigma=0.8, size=N)  # in INR
employees_baseline = np.random.randint(1, 15, size=N)
loan_access_baseline = np.random.choice([0, 1], size=N, p=[0.65, 0.35])
business_age_years = np.random.randint(1, 20, size=N)
owner_education = np.random.choice(
    ["No formal education", "Primary", "Secondary", "Graduate"],
    size=N, p=[0.2, 0.3, 0.3, 0.2]
)
sector = np.random.choice(
    ["Retail", "Manufacturing", "Services", "Agri-processing"],
    size=N, p=[0.35, 0.25, 0.25, 0.15]
)

# --- Endline revenue (treatment effect = ~15% uplift) ---
treatment_effect = 1.15
noise = np.random.normal(loc=1.0, scale=0.1, size=N)
endline_revenue = np.where(
    treatment == 1,
    baseline_revenue * treatment_effect * noise,
    baseline_revenue * noise
)

# --- Introducing messiness ---

# 1. Missing values in key columns (~5% each)
missing_mask_revenue = np.random.choice([True, False], size=N, p=[0.05, 0.95])
missing_mask_employees = np.random.choice([True, False], size=N, p=[0.04, 0.96])
missing_mask_education = np.random.choice([True, False], size=N, p=[0.06, 0.94])

endline_revenue = endline_revenue.astype(float)
endline_revenue[missing_mask_revenue] = np.nan

employees_baseline = employees_baseline.astype(float)
employees_baseline[missing_mask_employees] = np.nan

owner_education = list(owner_education)
for i in range(N):
    if missing_mask_education[i]:
        owner_education[i] = None

# 2. Inconsistent formatting in categorical column
sector = list(sector)
for i in range(N):
    if np.random.rand() < 0.08:
        sector[i] = sector[i].upper()   # e.g. "RETAIL" vs "Retail"
    elif np.random.rand() < 0.05:
        sector[i] = sector[i].lower()   # e.g. "retail"

# 3. One column with mixed date formats (survey date)
def random_date():
    day = np.random.randint(1, 28)
    month = np.random.randint(1, 13)
    year = np.random.choice([2022, 2023])
    fmt = np.random.choice(["dmy_slash", "mdy_slash", "iso"])
    if fmt == "dmy_slash":
        return f"{day:02d}/{month:02d}/{year}"
    elif fmt == "mdy_slash":
        return f"{month:02d}/{day:02d}/{year}"
    else:
        return f"{year}-{month:02d}-{day:02d}"

survey_date = [random_date() for _ in range(N)]

# --- Building DataFrame ---
df = pd.DataFrame({
    "business_id": business_ids,
    "state": state_col,
    "district": district_col,
    "treatment": treatment,
    "sector": sector,
    "business_age_years": business_age_years,
    "owner_education": owner_education,
    "loan_access_baseline": loan_access_baseline,
    "employees_baseline": employees_baseline,
    "baseline_revenue_inr": baseline_revenue.round(2),
    "endline_revenue_inr": endline_revenue.round(2),
    "survey_date": survey_date
})

# Saving raw(messy) data
df.to_csv("data/raw/msme_survey_raw.csv", index=False)
print(f"Raw dataset saved: {len(df)} rows, {df.shape[1]} columns")
print(f"Duplicated IDs introduced: {df['business_id'].duplicated().sum()}")
print(f"Missing endline revenue: {df['endline_revenue_inr'].isna().sum()}")