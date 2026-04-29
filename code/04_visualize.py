"""
04_visualize.py
---------------
Generates all publication-quality charts for the Women-Led MSME Impact
Evaluation pipeline. Reads from outputs/ CSVs produced by 03_analysis.py
and from the clean dataset produced by 02_clean_data.py.

Charts produced:
    Fig 1 — Revenue distribution: treatment vs control (kernel density)
    Fig 2 — Balance check: baseline characteristics (dot plot)
    Fig 3 — Regression coefficient plot (main OLS results)
    Fig 4 — Heterogeneity analysis: treatment effect by subgroup
    Fig 5 — Sector composition by treatment arm (stacked bar)
    Fig 6 — Revenue growth rate distribution by treatment arm (violin)

All figures saved to outputs/figures/ as high-resolution PNGs.
A combined summary figure (Fig 7) tiles the four core charts.

Author: Valentina Sharma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEAN_PATH   = "data/clean/msme_survey_clean.csv"
OUTPUT_DIR   = "outputs"
FIGURES_DIR  = "outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df         = pd.read_csv(CLEAN_PATH)
df_reg     = df[df["endline_revenue_missing"] == 0].copy()
balance_df = pd.read_csv(f"{OUTPUT_DIR}/balance_check.csv")
reg_df     = pd.read_csv(f"{OUTPUT_DIR}/regression_results.csv", index_col=0)
hetero_df  = pd.read_csv(f"{OUTPUT_DIR}/heterogeneity_results.csv")

# ── Design system ─────────────────────────────────────────────────────────────
# Palette: consistent across all charts
TREATMENT_COLOR = "#2166AC"   # deep blue  — treatment arm
CONTROL_COLOR   = "#D6604D"   # muted red  — control arm
ACCENT_COLOR    = "#4DAC26"   # green      — significant results
NEUTRAL_COLOR   = "#878787"   # grey       — non-significant / reference
BACKGROUND      = "#FAFAFA"
GRID_COLOR      = "#E8E8E8"

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "figure.facecolor":  BACKGROUND,
    "axes.facecolor":    BACKGROUND,
    "axes.edgecolor":    "#CCCCCC",
    "axes.linewidth":    0.8,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BACKGROUND,
})

def add_source_note(ax, note="Synthetic data for illustrative purposes. Pipeline: Women-Led MSME IE, 2024."):
    """Add a small source note at the bottom of any axis."""
    ax.annotate(
        note,
        xy=(0, -0.13), xycoords="axes fraction",
        fontsize=7, color="#999999", ha="left"
    )

def save_fig(fig, filename):
    path = f"{FIGURES_DIR}/{filename}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Revenue Distribution: Treatment vs Control (KDE + rug)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Revenue distribution — KDE plot")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Fig 1 — Revenue Distribution: Treatment vs Control",
    fontsize=14, fontweight="bold", y=1.01
)

for ax, (col, label) in zip(axes, [
    ("baseline_revenue_inr", "Baseline Revenue (INR)"),
    ("endline_revenue_inr",  "Endline Revenue (INR)")
]):
    for grp, color, name in [
        (1, TREATMENT_COLOR, "Treatment"),
        (0, CONTROL_COLOR,   "Control")
    ]:
        vals = df_reg[df_reg["treatment"] == grp][col].dropna()
        sns.kdeplot(
            vals / 1_000,          # display in thousands
            ax=ax, color=color, label=name,
            linewidth=2.2, fill=True, alpha=0.18
        )
        ax.axvline(
            vals.median() / 1_000,
            color=color, linestyle="--", linewidth=1.4, alpha=0.75
        )

    ax.set_xlabel(f"{label} (₹ thousands)", labelpad=6)
    ax.set_ylabel("Density")
    ax.set_title(label.split("(")[0].strip())
    ax.legend(frameon=True, framealpha=0.9)
    add_source_note(ax)

axes[0].set_title("Baseline Revenue (pre-intervention)")
axes[1].set_title("Endline Revenue (post-intervention)")

fig.tight_layout()
save_fig(fig, "fig1_revenue_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Balance Check: Dot Plot
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 2] Balance check — dot plot")

# Normalise means for display (show standardised difference)
balance_df = balance_df.copy()
balance_df["std_diff"] = balance_df["difference"] / (
    (balance_df["mean_treatment"] + balance_df["mean_control"]) / 2
).replace(0, np.nan) * 100   # as % of average

var_labels = {
    "baseline_revenue_inr":    "Baseline Revenue",
    "employees_baseline":      "Employees (baseline)",
    "business_age_years":      "Business Age (years)",
    "owner_education_numeric": "Education Level (numeric)",
    "loan_access_baseline":    "Loan Access (baseline)"
}
balance_df["var_label"] = balance_df["variable"].map(var_labels).fillna(balance_df["variable"])

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.suptitle(
    "Fig 2 — Baseline Balance Check: Treatment vs Control Groups",
    fontsize=14, fontweight="bold"
)

colors = [ACCENT_COLOR if b == "YES ✓" else "#E8474C" for b in balance_df["balanced"]]
bars = ax.barh(
    balance_df["var_label"],
    balance_df["std_diff"],
    color=colors, height=0.55, zorder=3, edgecolor="white", linewidth=0.5
)

ax.axvline(0, color="#444444", linewidth=1.2, zorder=4)
ax.axvline(-10, color=NEUTRAL_COLOR, linewidth=0.9, linestyle=":", zorder=2)
ax.axvline( 10, color=NEUTRAL_COLOR, linewidth=0.9, linestyle=":", zorder=2)
ax.fill_betweenx([-0.5, len(balance_df) - 0.5], -10, 10,
                  alpha=0.07, color=ACCENT_COLOR, zorder=1)

# p-value annotations
for i, (_, row) in enumerate(balance_df.iterrows()):
    x_pos = row["std_diff"] + (1.5 if row["std_diff"] >= 0 else -1.5)
    ha    = "left" if row["std_diff"] >= 0 else "right"
    star  = "***" if row["p_value"] < 0.01 else ("**" if row["p_value"] < 0.05 else ("*" if row["p_value"] < 0.10 else ""))
    label = f"p={row['p_value']:.3f}{star}"
    ax.text(x_pos, i, label, va="center", ha=ha, fontsize=8.5, color="#444444")

# Legend
patch_balanced   = mpatches.Patch(color=ACCENT_COLOR,  label="Balanced (p > 0.10)")
patch_unbalanced = mpatches.Patch(color="#E8474C",     label="Imbalanced (p ≤ 0.10)")
ax.legend(handles=[patch_balanced, patch_unbalanced], loc="lower right", frameon=True)

ax.set_xlabel("Normalised Difference (% of group average)")
ax.set_title("Shaded band = ±10% — values within band indicate good balance", fontsize=9, color="#666666")
ax.grid(axis="x", zorder=0)
add_source_note(ax)
fig.tight_layout()
save_fig(fig, "fig2_balance_check.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Regression Coefficient Plot (main OLS)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 3] Regression coefficient plot")

# Filter to meaningful covariates (exclude intercept and state FEs for readability)
plot_vars = reg_df[
    ~reg_df.index.str.startswith("state_fe_") &
    (reg_df.index != "Intercept")
].copy()

label_map = {
    "treatment":            "Treatment\n(key variable)",
    "log_baseline_revenue": "Log Baseline Revenue",
    "employees_baseline":   "Employees (baseline)",
    "loan_access_baseline": "Loan Access (baseline)",
    "business_age_years":   "Business Age (years)"
}
plot_vars["label"] = plot_vars.index.map(label_map).fillna(plot_vars.index)
plot_vars = plot_vars.iloc[::-1]   # flip for horizontal plot (treatment on top)

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.suptitle(
    "Fig 3 — OLS Regression Coefficients: Determinants of Log Endline Revenue",
    fontsize=14, fontweight="bold"
)

for i, (idx, row) in enumerate(plot_vars.iterrows()):
    is_treatment = (idx == "treatment")
    is_sig       = row["p_value"] < 0.10
    pt_color     = TREATMENT_COLOR if is_treatment else (ACCENT_COLOR if is_sig else NEUTRAL_COLOR)
    pt_size      = 130 if is_treatment else 80

    # CI line
    ax.plot(
        [row["ci_lower"], row["ci_upper"]], [i, i],
        color=pt_color, linewidth=2.2 if is_treatment else 1.5,
        solid_capstyle="round", zorder=3
    )
    # Point estimate
    ax.scatter(row["coefficient"], i, color=pt_color, s=pt_size, zorder=4,
               edgecolors="white", linewidths=0.7)

    # Significance stars
    star = row.get("significant", "")
    if star:
        ax.text(row["ci_upper"] + 0.01, i, star, va="center",
                fontsize=10, color=pt_color, fontweight="bold")

ax.axvline(0, color="#333333", linewidth=1.1, linestyle="--", zorder=2)
ax.set_yticks(range(len(plot_vars)))
ax.set_yticklabels(plot_vars["label"], fontsize=10)
ax.set_xlabel("Coefficient estimate (95% CI) — Robust (HC3) standard errors")

# Highlight treatment row
ax.axhspan(
    len(plot_vars) - 1 - 0.4,
    len(plot_vars) - 1 + 0.4,
    alpha=0.07, color=TREATMENT_COLOR, zorder=1
)

# Annotation box for treatment effect
t_row = reg_df.loc["treatment"]
effect_pct = round((np.exp(t_row["coefficient"]) - 1) * 100, 1)
ax.annotate(
    f"Treatment effect: +{effect_pct}% revenue\n(p = {t_row['p_value']:.4f})",
    xy=(t_row["coefficient"], len(plot_vars) - 1),
    xytext=(t_row["coefficient"] + 0.06, len(plot_vars) - 1 - 1.1),
    fontsize=9, color=TREATMENT_COLOR,
    arrowprops=dict(arrowstyle="->", color=TREATMENT_COLOR, lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor=TREATMENT_COLOR, alpha=0.9)
)

legend_handles = [
    mpatches.Patch(color=TREATMENT_COLOR, label="Treatment (key variable)"),
    mpatches.Patch(color=ACCENT_COLOR,    label="Significant covariate (p<0.10)"),
    mpatches.Patch(color=NEUTRAL_COLOR,   label="Not significant"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=9)
add_source_note(ax)
fig.tight_layout()
save_fig(fig, "fig3_regression_coefficients.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Heterogeneity Analysis: Treatment Effect by Subgroup
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 4] Heterogeneity analysis")

# Build a tidy subgroup effects table for plotting
#   For each subgroup: effect for base group, effect for subgroup (base + interaction)
subgroup_plot = []
for _, row in hetero_df.iterrows():
    base  = row["base_treatment_effect"]
    inter = row["interaction_coef"]
    p_int = row["p_interaction"]
    grp   = row["subgroup"]

    if grp == "loan_access_baseline":
        subgroup_plot.append({"group": "No prior loan access", "effect": base,         "sig": ""})
        subgroup_plot.append({"group": "Has prior loan access", "effect": base + inter, "sig": "**" if p_int < 0.05 else ("*" if p_int < 0.10 else "")})
    elif grp == "owner_education_graduate":
        subgroup_plot.append({"group": "Non-graduate owner",   "effect": base,         "sig": ""})
        subgroup_plot.append({"group": "Graduate owner",       "effect": base + inter, "sig": "**" if p_int < 0.05 else ("*" if p_int < 0.10 else "")})

spdf = pd.DataFrame(subgroup_plot)
spdf["effect_pct"] = (np.exp(spdf["effect"]) - 1) * 100

colors_het = [CONTROL_COLOR, TREATMENT_COLOR, CONTROL_COLOR, TREATMENT_COLOR]

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle(
    "Fig 4 — Heterogeneity Analysis: Treatment Effect by Subgroup",
    fontsize=14, fontweight="bold"
)

bars = ax.barh(
    spdf["group"], spdf["effect_pct"],
    color=colors_het, height=0.5, edgecolor="white", linewidth=0.6, zorder=3
)

ax.axvline(0, color="#333333", linewidth=1.1, linestyle="--", zorder=2)

# Add value labels
for bar, (_, row) in zip(bars, spdf.iterrows()):
    w = bar.get_width()
    ax.text(
        w + (0.3 if w >= 0 else -0.3),
        bar.get_y() + bar.get_height() / 2,
        f"{w:+.1f}%{row['sig']}",
        va="center", ha="left" if w >= 0 else "right",
        fontsize=9.5, fontweight="bold", color="#333333"
    )

# Divider between subgroup panels
ax.axhline(1.5, color="#CCCCCC", linewidth=1, linestyle="-")
ax.text(-0.5, 3.05, "Loan Access Subgroup", fontsize=9, color="#555555", style="italic")
ax.text(-0.5, 1.05, "Education Subgroup",   fontsize=9, color="#555555", style="italic")

ax.set_xlabel("Implied % change in endline revenue (treatment vs control)")
ax.set_title("Interaction terms from OLS with state FE and baseline controls", fontsize=9, color="#666666")
add_source_note(ax)
fig.tight_layout()
save_fig(fig, "fig4_heterogeneity.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Sector Composition by Treatment Arm (100% stacked bar)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 5] Sector composition")

sector_dist = pd.read_csv(f"{OUTPUT_DIR}/sector_distribution.csv", index_col=0)
# Ensure we have 0 and 1 columns
if "0" in sector_dist.columns:
    sector_dist = sector_dist.rename(columns={"0": 0, "1": 1})
sector_pct = sector_dist[[0, 1]].div(sector_dist[[0, 1]].sum(axis=0), axis=1) * 100

sector_colors = ["#2166AC", "#D6604D", "#4DAC26", "#FDAE61"]
sectors = sector_pct.index.tolist()

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle(
    "Fig 5 — Sector Composition by Treatment Arm",
    fontsize=14, fontweight="bold"
)

x       = np.array([0, 1])
bottoms = np.zeros(2)

for i, sector in enumerate(sectors):
    vals = sector_pct.loc[sector, [0, 1]].values.astype(float)
    bars = ax.bar(x, vals, bottom=bottoms, color=sector_colors[i],
                  label=sector, width=0.45, edgecolor="white", linewidth=0.8)
    # Center label inside bar if tall enough
    for j, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 5:
            ax.text(x[j], b + v / 2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
    bottoms += vals

ax.set_xticks([0, 1])
ax.set_xticklabels(["Control", "Treatment"], fontsize=11)
ax.set_ylabel("Share of firms (%)")
ax.set_ylim(0, 108)
ax.legend(loc="upper right", frameon=True, title="Sector", title_fontsize=9)
ax.grid(axis="y", zorder=0)
add_source_note(ax)
fig.tight_layout()
save_fig(fig, "fig5_sector_composition.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Revenue Growth Rate Distribution (Violin + Box)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 6] Revenue growth rate violin plot")

df_violin = df_reg[["treatment", "revenue_growth_rate"]].dropna().copy()
df_violin["arm"] = df_violin["treatment"].map({1: "Treatment", 0: "Control"})
# Remove extreme outliers for display only
q1, q99 = df_violin["revenue_growth_rate"].quantile([0.01, 0.99])
df_violin = df_violin[df_violin["revenue_growth_rate"].between(q1, q99)]

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.suptitle(
    "Fig 6 — Revenue Growth Rate Distribution: Treatment vs Control",
    fontsize=14, fontweight="bold"
)

vp = sns.violinplot(
    data=df_violin, x="arm", y="revenue_growth_rate",
    palette={"Treatment": TREATMENT_COLOR, "Control": CONTROL_COLOR},
    inner="box", linewidth=1.2, ax=ax,
    order=["Control", "Treatment"]
)

ax.axhline(0, color="#333333", linewidth=1.1, linestyle="--", alpha=0.6)

# Annotate medians
for i, grp in enumerate(["Control", "Treatment"]):
    med = df_violin[df_violin["arm"] == grp]["revenue_growth_rate"].median()
    ax.text(i, med + 0.015, f"Median: {med:.1%}",
            ha="center", fontsize=9, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=TREATMENT_COLOR if grp == "Treatment" else CONTROL_COLOR,
                      alpha=0.85, edgecolor="none"))

ax.set_ylabel("Revenue Growth Rate (endline/baseline − 1)")
ax.set_xlabel("")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_title("Inner box shows IQR and median; width shows density of observations",
             fontsize=9, color="#666666")
add_source_note(ax)
fig.tight_layout()
save_fig(fig, "fig6_revenue_growth_violin.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Combined Summary Figure (2×2 tile of core charts)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 7] Combined summary figure")

from matplotlib.image import imread

panel_files = [
    ("fig1_revenue_distribution.png", "Revenue Distribution"),
    ("fig2_balance_check.png",        "Balance Check"),
    ("fig3_regression_coefficients.png", "OLS Regression"),
    ("fig4_heterogeneity.png",        "Heterogeneity Analysis"),
]

fig = plt.figure(figsize=(18, 13))
fig.suptitle(
    "Women-Led MSME Impact Evaluation — Key Results Summary",
    fontsize=17, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08)

for idx, (fname, title) in enumerate(panel_files):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    try:
        img = imread(f"{FIGURES_DIR}/{fname}")
        ax.imshow(img, aspect="auto")
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"[{fname}]", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#999999")
    ax.axis("off")
    ax.set_title(f"({chr(65 + idx)}) {title}", fontsize=12,
                 fontweight="bold", pad=6, loc="left")

fig.savefig(f"{FIGURES_DIR}/fig7_summary_panel.png", dpi=200)
plt.close(fig)
print(f"  ✓ Saved: {FIGURES_DIR}/fig7_summary_panel.png")


# ══════════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("VISUALISATION COMPLETE — ALL FIGURES SAVED")
print("=" * 60)
figs = [
    "fig1_revenue_distribution.png",
    "fig2_balance_check.png",
    "fig3_regression_coefficients.png",
    "fig4_heterogeneity.png",
    "fig5_sector_composition.png",
    "fig6_revenue_growth_violin.png",
    "fig7_summary_panel.png",
]
for f in figs:
    print(f"  ✓ {FIGURES_DIR}/{f}")
print(f"\nNext step: open notebooks/full_pipeline.ipynb for the unified walkthrough.")