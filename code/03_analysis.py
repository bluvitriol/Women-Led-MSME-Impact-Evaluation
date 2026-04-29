"""
03_analysis.py
--------------
Runs the full impact evaluation analysis on the clean MSME dataset.

Analysis modules:
    1.  Descriptive statistics — full sample and by treatment group
    2.  Balance check — are treatment and control groups similar at baseline?
    3.  Main regression — OLS: log_endline_revenue ~ treatment + controls
    4.  Heterogeneity analysis — treatment effect by education and loan access
    5.  Save all outputs as CSVs for replication

All outputs saved to outputs/ for use in 04_visualize.py and for
push-button replication (no manual steps required).

Author: Valentina Sharma
"""

import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
CLEAN_PATH  = "data/clean/msme_survey_clean.csv"
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load clean data ───────────────────────────────────────────────────────────
df = pd.read_csv(CLEAN_PATH)

# For regression: drop rows with missing primary outcome
df_reg = df[df["endline_revenue_missing"] == 0].copy()

print(f"Full sample:            {len(df)} observations")
print(f"Regression sample:      {len(df_reg)} observations (dropped {len(df) - len(df_reg)} missing endline)")
print(f"Treatment group:        {df_reg['treatment'].sum()} firms")
print(f"Control group:          {(df_reg['treatment'] == 0).sum()} firms\n")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Descriptive Statistics
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MODULE 1: DESCRIPTIVE STATISTICS")
print("=" * 60)

desc_cols = [
    "baseline_revenue_inr", "endline_revenue_inr",
    "revenue_growth_rate", "employees_baseline",
    "loan_access_baseline", "business_age_years",
    "owner_education_numeric"
]

# Overall descriptives
desc_overall = df[desc_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
desc_overall.columns = ["n", "mean", "std", "min", "median", "max"]
desc_overall = desc_overall.round(3)

print("\nOverall descriptive statistics:")
print(desc_overall.to_string())

# By treatment group
desc_by_treatment = df.groupby("treatment")[desc_cols].agg(["mean", "std"]).round(3)
print("\nMeans by treatment group (1=treatment, 0=control):")
print(df.groupby("treatment")[desc_cols].mean().round(3).to_string())

# Sector distribution
sector_dist = df.groupby(["sector", "treatment"]).size().unstack(fill_value=0)
sector_dist["total"] = sector_dist.sum(axis=1)
print("\nSector distribution by treatment arm:")
print(sector_dist.to_string())

# Save
desc_overall.to_csv(f"{OUTPUT_DIR}/desc_stats_overall.csv")
df.groupby("treatment")[desc_cols].mean().round(3).to_csv(f"{OUTPUT_DIR}/desc_stats_by_treatment.csv")
sector_dist.to_csv(f"{OUTPUT_DIR}/sector_distribution.csv")
print(f"\n→ Saved descriptive stats to {OUTPUT_DIR}/")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Balance Check
# ════════════════════════════════════════════════════════════════════════════
# Test whether baseline characteristics are similar across treatment arms.
# We run two-sample t-tests for continuous variables and chi-square for categorical.
# A p-value > 0.10 indicates no significant imbalance (good randomization).

print("\n" + "=" * 60)
print("MODULE 2: BALANCE CHECK (BASELINE CHARACTERISTICS)")
print("=" * 60)

balance_vars_continuous = [
    "baseline_revenue_inr",
    "employees_baseline",
    "business_age_years",
    "owner_education_numeric"
]

treatment_group = df[df["treatment"] == 1]
control_group   = df[df["treatment"] == 0]

balance_results = []

for var in balance_vars_continuous:
    t_vals = treatment_group[var].dropna()
    c_vals = control_group[var].dropna()
    t_stat, p_val = stats.ttest_ind(t_vals, c_vals, equal_var=False)  # Welch's t-test
    balance_results.append({
        "variable":       var,
        "mean_treatment": round(t_vals.mean(), 3),
        "mean_control":   round(c_vals.mean(), 3),
        "difference":     round(t_vals.mean() - c_vals.mean(), 3),
        "t_statistic":    round(t_stat, 3),
        "p_value":        round(p_val, 4),
        "balanced":       "YES ✓" if p_val > 0.10 else "NO ✗"
    })

# Loan access — chi-square test (binary variable)
loan_crosstab = pd.crosstab(df["treatment"], df["loan_access_baseline"])
chi2, p_chi, _, _ = stats.chi2_contingency(loan_crosstab)
balance_results.append({
    "variable":       "loan_access_baseline",
    "mean_treatment": round(treatment_group["loan_access_baseline"].mean(), 3),
    "mean_control":   round(control_group["loan_access_baseline"].mean(), 3),
    "difference":     round(treatment_group["loan_access_baseline"].mean() - control_group["loan_access_baseline"].mean(), 3),
    "t_statistic":    round(chi2, 3),
    "p_value":        round(p_chi, 4),
    "balanced":       "YES ✓" if p_chi > 0.10 else "NO ✗"
})

balance_df = pd.DataFrame(balance_results)
print("\nBalance test results (Welch's t-test for continuous; chi-square for binary):")
print(balance_df.to_string(index=False))

balance_df.to_csv(f"{OUTPUT_DIR}/balance_check.csv", index=False)
print(f"\n→ Saved balance check to {OUTPUT_DIR}/balance_check.csv")

# Summarise
n_unbalanced = (balance_df["balanced"] == "NO ✗").sum()
if n_unbalanced == 0:
    print("\n✓ All baseline characteristics balanced across treatment arms.")
else:
    print(f"\n⚠ {n_unbalanced} variable(s) show imbalance — consider including as controls in regression.")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 3 — Main OLS Regression
# ════════════════════════════════════════════════════════════════════════════
# Specification: log(endline_revenue) = α + β·treatment + γ·log(baseline_revenue)
#                + δ·controls + state_FE + ε
#
# β is the ITT (Intent-to-Treat) estimate — the average effect of being
# assigned to treatment on log endline revenue.
# Interpreting β: a one-unit change in treatment → β × 100% change in revenue.

print("\n" + "=" * 60)
print("MODULE 3: MAIN OLS REGRESSION")
print("=" * 60)

# Build state FE variable names (dummies created in cleaning script)
state_fe_cols = [c for c in df_reg.columns if c.startswith("state_fe_")]
fe_formula_part = " + ".join(state_fe_cols) if state_fe_cols else ""

formula = (
    "log_endline_revenue ~ treatment "
    "+ log_baseline_revenue "
    "+ employees_baseline "
    "+ loan_access_baseline "
    "+ business_age_years "
    + (f"+ {fe_formula_part}" if fe_formula_part else "")
)

print(f"\nRegression formula:\n  {formula}\n")

model = smf.ols(formula=formula, data=df_reg).fit(cov_type="HC3")  # HC3 = heteroskedasticity-robust SEs

print(model.summary())

# Extract clean results table
results_table = pd.DataFrame({
    "coefficient":  model.params.round(4),
    "std_error":    model.bse.round(4),
    "t_statistic":  model.tvalues.round(3),
    "p_value":      model.pvalues.round(4),
    "ci_lower":     model.conf_int()[0].round(4),
    "ci_upper":     model.conf_int()[1].round(4),
    "significant":  model.pvalues.apply(lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "")))
})

results_table.to_csv(f"{OUTPUT_DIR}/regression_results.csv")

# Print key finding
treatment_coef = model.params["treatment"]
treatment_pval = model.pvalues["treatment"]
treatment_effect_pct = round((np.exp(treatment_coef) - 1) * 100, 2)

print(f"\n── KEY FINDING ──────────────────────────────────────")
print(f"Treatment coefficient (β):  {treatment_coef:.4f}")
print(f"p-value:                    {treatment_pval:.4f}")
print(f"Implied revenue effect:     +{treatment_effect_pct}% endline revenue")
print(f"R-squared:                  {model.rsquared:.4f}")
print(f"N (regression sample):      {int(model.nobs)}")
print(f"────────────────────────────────────────────────────\n")
print(f"→ Saved regression results to {OUTPUT_DIR}/regression_results.csv")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Heterogeneity Analysis
# ════════════════════════════════════════════════════════════════════════════
# Does the treatment effect differ by:
#   (a) Loan access at baseline?
#   (b) Owner education level?
# Method: interaction terms (treatment × subgroup indicator)

print("\n" + "=" * 60)
print("MODULE 4: HETEROGENEITY ANALYSIS")
print("=" * 60)

hetero_results = []

# 4a — By loan access at baseline
print("\n(a) Heterogeneity by baseline loan access:")
fe_part = f"+ {fe_formula_part}" if fe_formula_part else ""

formula_loan = (
    "log_endline_revenue ~ treatment * loan_access_baseline "
    "+ log_baseline_revenue + employees_baseline "
    f"+ business_age_years {fe_part}"
)
model_loan = smf.ols(formula=formula_loan, data=df_reg).fit(cov_type="HC3")

coef_no_loan  = model_loan.params.get("treatment", np.nan)
coef_loan_int = model_loan.params.get("treatment:loan_access_baseline", np.nan)

print(f"  Treatment effect (no prior loan):     {coef_no_loan:.4f}  (p={model_loan.pvalues.get('treatment', np.nan):.4f})")
print(f"  Additional effect for loan holders:   {coef_loan_int:.4f}  (p={model_loan.pvalues.get('treatment:loan_access_baseline', np.nan):.4f})")

hetero_results.append({
    "subgroup": "loan_access_baseline",
    "base_treatment_effect": round(coef_no_loan, 4),
    "interaction_coef": round(coef_loan_int, 4),
    "p_interaction": round(model_loan.pvalues.get("treatment:loan_access_baseline", np.nan), 4)
})

# 4b — By education (graduate vs non-graduate)
print("\n(b) Heterogeneity by owner education (graduate vs non-graduate):")
df_reg["is_graduate"] = (df_reg["owner_education_numeric"] >= 3).astype(int)

formula_edu = (
    "log_endline_revenue ~ treatment * is_graduate "
    "+ log_baseline_revenue + employees_baseline "
    f"+ business_age_years {fe_part}"
)
model_edu = smf.ols(formula=formula_edu, data=df_reg.dropna(subset=["owner_education_numeric"])).fit(cov_type="HC3")

coef_non_grad  = model_edu.params.get("treatment", np.nan)
coef_grad_int  = model_edu.params.get("treatment:is_graduate", np.nan)

print(f"  Treatment effect (non-graduate owners): {coef_non_grad:.4f}  (p={model_edu.pvalues.get('treatment', np.nan):.4f})")
print(f"  Additional effect for graduates:        {coef_grad_int:.4f}  (p={model_edu.pvalues.get('treatment:is_graduate', np.nan):.4f})")

hetero_results.append({
    "subgroup": "owner_education_graduate",
    "base_treatment_effect": round(coef_non_grad, 4),
    "interaction_coef": round(coef_grad_int, 4),
    "p_interaction": round(model_edu.pvalues.get("treatment:is_graduate", np.nan), 4)
})

hetero_df = pd.DataFrame(hetero_results)
hetero_df.to_csv(f"{OUTPUT_DIR}/heterogeneity_results.csv", index=False)
print(f"\n→ Saved heterogeneity results to {OUTPUT_DIR}/heterogeneity_results.csv")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY — All outputs produced
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE — OUTPUTS SAVED")
print("=" * 60)
outputs = [
    "desc_stats_overall.csv",
    "desc_stats_by_treatment.csv",
    "sector_distribution.csv",
    "balance_check.csv",
    "regression_results.csv",
    "heterogeneity_results.csv"
]
for f in outputs:
    print(f"  ✓ {OUTPUT_DIR}/{f}")

print(f"\nNext step: run 04_visualize.py to generate charts from these outputs.")