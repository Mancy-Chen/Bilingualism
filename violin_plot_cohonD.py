import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, chi2
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

sns.set_context("talk", font_scale=1.4)

# ============================
# CONFIG
# ============================

# Switch between non-bias-corrected and bias-corrected BAG
USE_BIAS_CORRECTED = True  # True = use bias-corrected deltas; False = use raw predicted ages

true_age_col = "Age"
group_col = "group"
subj_col = "MRI code"  # Subject ID
group_order = ["bilinguals", "translators", "interpreters"]

if USE_BIAS_CORRECTED:
    # Bias-corrected results
    merged_path = "/data/projects/CSC/code/Bilingualism/09_output_compare/brainpad_results.xlsx"
    bag_cols = [
        "delta_cv5_Predicted_age_non_BC_Brainage",
        "delta_cv5_Predicted_age_non_BC_BrainageR",
        "delta_cv5_Predicted_age_non_BC_Deepbrainnet",
        "delta_cv5_Predicted_age_non_BC_Pyment",
        "delta_cv5_Predicted_age_non_BC_BRAID_WM",
        "delta_cv5_Predicted_age_non_BC_BRAID_GM",
    ]

    pretty_names = {
        "delta_cv5_Predicted_age_non_BC_Brainage": "BrainAge",
        "delta_cv5_Predicted_age_non_BC_BrainageR": "BrainageR",
        "delta_cv5_Predicted_age_non_BC_Deepbrainnet": "DeepBrainNet",
        "delta_cv5_Predicted_age_non_BC_Pyment": "Pyment",
        "delta_cv5_Predicted_age_non_BC_BRAID_WM": "BRAID WM",
        "delta_cv5_Predicted_age_non_BC_BRAID_GM": "BRAID GM",
    }

    fig_title = "Bias-corrected Brain Age Gap by Model and Group"
    out_png = "/data/projects/CSC/code/Bilingualism/09_output_compare/brainage_violin_plot_6models_BC_CohenD.png"

else:
    # Non-bias-corrected results
    merged_path = "/data/projects/CSC/code/Bilingualism/09_output_compare/BILCZE_brainpad_results.xlsx"
    bag_cols = [
        "Predicted_age_non_BC_Brainage",
        "Predicted_age_non_BC_BrainageR",
        "Predicted_age_non_BC_Deepbrainnet",
        "Predicted_age_non_BC_Pyment",
        "Predicted_age_non_BC_BRAID_WM",
        "Predicted_age_non_BC_BRAID_GM",
    ]

    pretty_names = {
        "Predicted_age_non_BC_Brainage": "BrainAge",
        "Predicted_age_non_BC_BrainageR": "BrainageR",
        "Predicted_age_non_BC_Deepbrainnet": "DeepBrainNet",
        "Predicted_age_non_BC_Pyment": "Pyment",
        "Predicted_age_non_BC_BRAID_WM": "BRAID WM",
        "Predicted_age_non_BC_BRAID_GM": "BRAID GM",
    }

    fig_title = "Brain Age Gap by Model and Group"
    out_png = "/data/projects/CSC/code/Bilingualism/09_output_compare/brainage_violin_plot_6models_CohenD.png"

# Multiple-comparison choices for the console stats
ANOVA_ACROSS_MODELS_METHOD = "holm"   # 'holm' (FWER) or 'fdr_bh'
PAIRS_ACROSS_MODELS_METHOD = "fdr_bh" # 'holm' or 'fdr_bh'

ANNOT_FONTSIZE = 20  # font size for brackets and Cohen's d labels

# ============================
# Load & reshape
# ============================

df = pd.read_excel(merged_path)

df_long = df.melt(
    id_vars=[subj_col, group_col],
    value_vars=bag_cols,
    var_name="Model",
    value_name="BrainAgeGap",
)

df_long["Model"] = (
    df_long["Model"]
    .str.replace(r"^BAG_", "", regex=True)
    .map(pretty_names)
)

df_long = df_long.dropna(subset=["BrainAgeGap", group_col, subj_col]).copy()
df_long = df_long[df_long[group_col].isin(group_order)].copy()

# desired model order
_alias_to_pretty = {
    "Brainage": "BrainAge",
    "BrainageR": "BrainageR",
    "Deepbrainnet": "DeepBrainNet",
    "Pyment": "Pyment",
    "BRAID_WM": "BRAID WM",
    "BRAID_GM": "BRAID GM",
    "Ours": "Our ML model",
}
desired_order_alias = [
    "Brainage",
    "BrainageR",
    "Deepbrainnet",
    "Pyment",
    "BRAID_WM",
    "BRAID_GM",
    "Ours",
]
desired_pretty = [_alias_to_pretty[a] for a in desired_order_alias]
present = set(df_long["Model"].unique())
model_order = [m for m in desired_pretty if m in present]

# ============================
# Mixed-effects omnibus (console)
# ============================

print("\n=== Mixed-Effects Omnibus Test (Group × Model) ===")
full = smf.mixedlm(
    "BrainAgeGap ~ C(group)*C(Model)",
    data=df_long,
    groups=df_long[subj_col],
)
fit_full = full.fit(reml=False)

reduced = smf.mixedlm(
    "BrainAgeGap ~ C(group) + C(Model)",
    data=df_long,
    groups=df_long[subj_col],
)
fit_red = reduced.fit(reml=False)

LR = 2 * (fit_full.llf - fit_red.llf)
df_diff = max(int(fit_full.df_modelwc - fit_red.df_modelwc), 1)
p_lrt = chi2.sf(LR, df_diff)
print(f"LR = {LR:.3f}, df = {df_diff}, p = {p_lrt:.6g}")
print("\nFull model summary:")
print(fit_full.summary())

# ============================
# Per-model ANOVA + Tukey
# ============================

def safe_groups_data(sub, groups):
    return [
        sub.loc[sub[group_col] == g, "BrainAgeGap"].dropna()
        for g in groups
        if g in set(sub[group_col])
    ]

anova_pvals = {}
tukey_pvals_by_model = {}

print("\n=== Per-Model One-Way ANOVA & Tukey ===")
for m in model_order:
    sub = df_long[df_long["Model"] == m].copy()
    present_groups = [g for g in group_order if g in set(sub[group_col])]
    if len(present_groups) < 2:
        print(f"{m}: <2 groups present, skipping.")
        continue

    g_dat = safe_groups_data(sub, present_groups)
    if sum(len(gv) > 0 for gv in g_dat) < 2:
        print(f"{m}: insufficient data for ANOVA.")
        continue

    f_stat, p_val = f_oneway(*g_dat)
    anova_pvals[m] = float(p_val)
    print(f"{m}: ANOVA p = {p_val:.6g}")

    tk = pairwise_tukeyhsd(
        endog=sub["BrainAgeGap"],
        groups=sub[group_col],
        alpha=0.05,
    )
    print(tk.summary())

    pairs = []
    for row in tk._results_table.data[1:]:
        g1, g2, meandiff, p_adj, lower, upper, reject = row
        g1, g2 = str(g1), str(g2)
        if g1 in group_order and g2 in group_order:
            pairs.append((g1, g2, float(p_adj)))
    tukey_pvals_by_model[m] = pairs

# cross-model corrections (console only)
if anova_pvals:
    models_list = list(anova_pvals.keys())
    anova_raw = [anova_pvals[m] for m in models_list]
    rej_anova, p_anova_corr, _, _ = multipletests(
        anova_raw, method=ANOVA_ACROSS_MODELS_METHOD
    )
    anova_p_corr = dict(zip(models_list, p_anova_corr))
else:
    anova_p_corr = {}

all_keys, all_p = [], []
for m, pairs in tukey_pvals_by_model.items():
    for (g1, g2, p) in pairs:
        all_keys.append((m, g1, g2))
        all_p.append(p)

if all_p:
    rej_pairs, p_pairs_corr, _, _ = multipletests(
        all_p, method=PAIRS_ACROSS_MODELS_METHOD
    )
    pair_p_corr = dict(zip(all_keys, p_pairs_corr))
    pair_reject = dict(zip(all_keys, rej_pairs))
else:
    pair_p_corr, pair_reject = {}, {}

# ============================
# Effect size: Cohen's d
# ============================

def cohen_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = x.mean(), y.mean()
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan
    return (m1 - m2) / sp

# ============================
# Plot helpers
# ============================

def p_to_stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

def hue_pos(x_index, hue_index, n_hue, width=0.8):
    return x_index - width / 2 + (hue_index + 0.5) * (width / n_hue)

def add_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], c="k", lw=1.0, clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=ANNOT_FONTSIZE,
        clip_on=False,
    )
    return y + h

# ============================
# Violin plot with brackets + Cohen's d
# ============================

plt.figure(figsize=(22, 10))
vi = sns.violinplot(
    data=df_long,
    x="Model",
    y="BrainAgeGap",
    hue=group_col,
    order=model_order,
    hue_order=group_order,
    split=False,
    inner="box",
    cut=0,
)
ax = plt.gca()
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_ylabel("Brain Age Gap (Predicted − Age)")
ax.set_title(fig_title)
ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax * 1.35)

n_hue = len(group_order)

for mi, model in enumerate(model_order):
    sub = df_long[df_long["Model"] == model].copy()
    if sub.empty:
        continue

    # Cohen's d: bilinguals vs translators+interpreters
    x_bi = sub.loc[sub[group_col] == "bilinguals", "BrainAgeGap"].dropna()
    x_prof = sub.loc[
        sub[group_col].isin(["translators", "interpreters"]), "BrainAgeGap"
    ].dropna()
    d_val = cohen_d(x_bi, x_prof)

    # vertical span
    yvals = sub["BrainAgeGap"].values
    ymax_m = np.nanmax(yvals) if yvals.size else 0.0
    ymin_m = np.nanmin(yvals) if yvals.size else 0.0
    span = (
        (ymax_m - ymin_m)
        if np.isfinite(ymax_m - ymin_m) and (ymax_m - ymin_m) > 0
        else 1.0
    )
    base_y = ymax_m + 0.05 * span
    step_h = 0.06 * span
    bar_h = 0.01 * span

    # brackets for significant Tukey contrasts (after global correction)
    used_levels = []
    pairs = tukey_pvals_by_model.get(model, [])
    for (g1, g2, p_tuk) in pairs:
        key = (model, g1, g2)
        if not pair_reject.get(key, False):
            continue
        p_corr = pair_p_corr.get(key, p_tuk)

        i1 = group_order.index(g1)
        i2 = group_order.index(g2)
        x1 = hue_pos(mi, i1, n_hue, width=0.8)
        x2 = hue_pos(mi, i2, n_hue, width=0.8)

        left, right = sorted([x1, x2])
        level = 0
        while any(
            (left <= L <= right) or (left <= R <= right) or (L <= left and right <= R)
            for (L, R, lvl) in used_levels
            if lvl == level
        ):
            level += 1
        used_levels.append((left, right, level))

        y = base_y + level * step_h
        add_bracket(ax, x1, x2, y, h=bar_h, text=p_to_stars(p_corr))

    # Cohen's d label above highest bracket
    if np.isfinite(d_val):
        max_level = max([lvl for _, _, lvl in used_levels], default=-1)
        d_y = base_y + (max_level + 1.8) * step_h
        ax.text(
            mi,
            d_y,
            f"d = {d_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=ANNOT_FONTSIZE,
        )

plt.tight_layout()
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\n📈 Saved annotated plot to: {out_png}")
