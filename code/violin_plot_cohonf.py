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

USE_BIAS_CORRECTED = True  # True = use bias-corrected deltas; False = use raw predicted ages

true_age_col = "Age"
group_col = "group"
subj_col = "MRI code"  # Subject ID
group_order = ["bilinguals", "translators", "interpreters"]

# Exploratory/descriptive choices
SHOW_ANOVA_ACROSS_MODELS_CORRECTION_IN_CONSOLE = True
ANOVA_ACROSS_MODELS_METHOD = "fdr_bh"  # 'fdr_bh' or 'holm'

ANNOT_FONTSIZE = 20

# Choose which omnibus effect size to show above each model:
# "f" (Cohen's f), "omega2", or "eta2"
OMNIBUS_EFFECT_TO_PLOT = "f"

# ============================
# Paths / columns
# ============================

if USE_BIAS_CORRECTED:
    merged_path = ".../Bilingualism/09_output_compare/brainpad_results.xlsx"
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
    out_png = ".../Bilingualism/09_output_compare/brainage_violin_plot_6models_BC_TukeyOnly_OmnibusEffect.png"
else:
    merged_path = ".../Bilingualism/09_output_compare/BILCZE_brainpad_results.xlsx"
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
    out_png = ".../Bilingualism/09_output_compare/brainage_violin_plot_6models_TukeyOnly_OmnibusEffect.png"

_alias_to_pretty = {
    "Brainage": "BrainAge",
    "BrainageR": "BrainageR",
    "Deepbrainnet": "DeepBrainNet",
    "Pyment": "Pyment",
    "BRAID_WM": "BRAID WM",
    "BRAID_GM": "BRAID GM",
}
desired_order_alias = ["Brainage", "BrainageR", "Deepbrainnet", "Pyment", "BRAID_WM", "BRAID_GM"]
desired_pretty = [_alias_to_pretty[a] for a in desired_order_alias]

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

df_long["Model"] = df_long["Model"].map(pretty_names).fillna(df_long["Model"])
df_long = df_long.dropna(subset=["BrainAgeGap", group_col, subj_col]).copy()
df_long = df_long[df_long[group_col].isin(group_order)].copy()

df_long[group_col] = pd.Categorical(df_long[group_col], categories=group_order, ordered=True)

present = set(df_long["Model"].unique())
model_order = [m for m in desired_pretty if m in present]

# ============================
# Mixed-effects omnibus (console)
# ============================

print("\n=== Mixed-Effects Omnibus Test (Group × Model) ===")
full = smf.mixedlm("BrainAgeGap ~ C(group)*C(Model)", data=df_long, groups=df_long[subj_col])
fit_full = full.fit(reml=False)

reduced = smf.mixedlm("BrainAgeGap ~ C(group) + C(Model)", data=df_long, groups=df_long[subj_col])
fit_red = reduced.fit(reml=False)

LR = 2 * (fit_full.llf - fit_red.llf)
df_diff = max(int(fit_full.df_modelwc - fit_red.df_modelwc), 1)
p_lrt = chi2.sf(LR, df_diff)
print(f"LR = {LR:.3f}, df = {df_diff}, p = {p_lrt:.6g}")
print("\nFull model summary:")
print(fit_full.summary())

# ============================
# Helpers: omnibus effect sizes for 1-way ANOVA (3 groups)
# ============================

def anova_effect_sizes_oneway(sub, groups):
    """Return eta2, omega2, and Cohen's f for one-way ANOVA."""
    vals_by_g = []
    for g in groups:
        v = sub.loc[sub[group_col] == g, "BrainAgeGap"].dropna().to_numpy(float)
        if len(v) > 0:
            vals_by_g.append(v)
    k = len(vals_by_g)
    if k < 2:
        return dict(eta2=np.nan, omega2=np.nan, f=np.nan)

    all_vals = np.concatenate(vals_by_g)
    N = len(all_vals)
    if N <= k:
        return dict(eta2=np.nan, omega2=np.nan, f=np.nan)

    grand = all_vals.mean()

    ss_between = 0.0
    ss_within = 0.0
    for v in vals_by_g:
        ss_between += len(v) * (v.mean() - grand) ** 2
        ss_within += ((v - v.mean()) ** 2).sum()

    ss_total = ss_between + ss_within
    df_between = k - 1
    df_within = N - k
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    omega2 = (
        (ss_between - df_between * ms_within) / (ss_total + ms_within)
        if np.isfinite(ms_within) and (ss_total + ms_within) > 0
        else np.nan
    )
    f = np.sqrt(eta2 / (1 - eta2)) if np.isfinite(eta2) and eta2 < 1 else np.nan

    return dict(eta2=eta2, omega2=omega2, f=f)

# ============================
# Per-model ANOVA + Tukey (within-model only)
# ============================

def safe_groups_data(sub, groups):
    return [sub.loc[sub[group_col] == g, "BrainAgeGap"].dropna() for g in groups if g in set(sub[group_col])]

anova_pvals = {}
tukey_by_model = {}
omnibus_by_model = {}  # model -> dict(eta2, omega2, f)

print("\n=== Per-Model One-Way ANOVA & Tukey (within-model only) ===")
for m in model_order:
    sub = df_long[df_long["Model"] == m].copy()
    present_groups = [g for g in group_order if g in set(sub[group_col])]
    if len(present_groups) < 2:
        print(f"{m}: <2 groups present, skipping.")
        continue

    # omnibus effect sizes
    omnibus_by_model[m] = anova_effect_sizes_oneway(sub, present_groups)

    g_dat = safe_groups_data(sub, present_groups)
    if sum(len(gv) > 0 for gv in g_dat) < 2:
        print(f"{m}: insufficient data for ANOVA.")
        continue

    f_stat, p_val = f_oneway(*g_dat)
    anova_pvals[m] = float(p_val)
    print(f"\n{m}: ANOVA p = {p_val:.6g} | eta2={omnibus_by_model[m]['eta2']:.3f} "
          f"omega2={omnibus_by_model[m]['omega2']:.3f} f={omnibus_by_model[m]['f']:.3f}")

    tk = pairwise_tukeyhsd(endog=sub["BrainAgeGap"], groups=sub[group_col], alpha=0.05)
    print(tk.summary())

    pairs = []
    for row in tk._results_table.data[1:]:
        g1, g2, meandiff, p_adj, lower, upper, reject = row
        g1, g2 = str(g1), str(g2)
        if (g1 in group_order) and (g2 in group_order):
            pairs.append((g1, g2, float(p_adj), bool(reject)))
    tukey_by_model[m] = pairs

# Optional: across-model correction for the 6 ANOVA p-values (console only)
if SHOW_ANOVA_ACROSS_MODELS_CORRECTION_IN_CONSOLE and anova_pvals:
    models_list = list(anova_pvals.keys())
    anova_raw = [anova_pvals[m] for m in models_list]
    rej, p_corr, _, _ = multipletests(anova_raw, method=ANOVA_ACROSS_MODELS_METHOD)
    print(f"\n=== ANOVA across-model correction ({ANOVA_ACROSS_MODELS_METHOD}) [console only] ===")
    for m, pr, pc, r in zip(models_list, anova_raw, p_corr, rej):
        print(f"{m:>12s}  p_raw={pr:.6g}  p_corr={pc:.6g}  reject={bool(r)}")

# ============================
# Plot helpers
# ============================

def p_to_stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

def hue_pos(x_index, hue_index, n_hue, width=0.8):
    return x_index - width / 2 + (hue_index + 0.5) * (width / n_hue)

def add_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], c="k", lw=1.0, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=ANNOT_FONTSIZE, clip_on=False)
    return y + h

# ============================
# Violin plot with Tukey brackets + omnibus effect size label
# ============================

plt.figure(figsize=(22, 10))
sns.violinplot(
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

# extend y-limits for annotations
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax * 1.35)

n_hue = len(group_order)

for mi, model in enumerate(model_order):
    sub = df_long[df_long["Model"] == model].copy()
    if sub.empty:
        continue

    # Omnibus effect size label (3-group)
    eff = omnibus_by_model.get(model, {})
    eff_val = eff.get(OMNIBUS_EFFECT_TO_PLOT, np.nan)

    # vertical span for bracket placement
    yvals = sub["BrainAgeGap"].to_numpy(float)
    ymax_m = np.nanmax(yvals) if yvals.size else 0.0
    ymin_m = np.nanmin(yvals) if yvals.size else 0.0
    span = (ymax_m - ymin_m) if np.isfinite(ymax_m - ymin_m) and (ymax_m - ymin_m) > 0 else 1.0
    base_y = ymax_m + 0.05 * span
    step_h = 0.06 * span
    bar_h = 0.01 * span

    # brackets: significant Tukey contrasts WITHIN model only
    used_levels = []
    pairs = tukey_by_model.get(model, [])
    for (g1, g2, p_tuk_adj, reject) in pairs:
        if not reject:
            continue

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
        add_bracket(ax, x1, x2, y, h=bar_h, text=p_to_stars(p_tuk_adj))

    # effect size label above the highest bracket
    if np.isfinite(eff_val):
        max_level = max([lvl for _, _, lvl in used_levels], default=-1)
        eff_y = base_y + (max_level + 1.8) * step_h

        if OMNIBUS_EFFECT_TO_PLOT == "f":
            label = f"f = {eff_val:.2f}"
        elif OMNIBUS_EFFECT_TO_PLOT == "omega2":
            label = f"ω² = {eff_val:.2f}"
        else:
            label = f"η² = {eff_val:.2f}"

        ax.text(mi, eff_y, label, ha="center", va="bottom", fontsize=ANNOT_FONTSIZE)

plt.tight_layout()
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\n📈 Saved annotated plot to: {out_png}")
