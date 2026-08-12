#!/usr/bin/env python3
"""Six-model BAG screening, multiplicity correction, and violin plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import chi2, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

INPUT_XLSX = Path("input/brainpad_results_deidentified.xlsx")
OUTPUT_DIR = Path("output")
USE_CORRECTED = True

SUBJECT_COL = "MRI code"
GROUP_COL = "group"
GROUP_ORDER = ["bilinguals", "translators", "interpreters"]
MULTIPLICITY_METHOD = "fdr_bh"

MODEL_NAMES = [
    "BrainAge",
    "BrainAgeR",
    "DeepBrainNet",
    "Pyment",
    "BRAID_WM",
    "BRAID_GM",
]

PRETTY_NAMES = {
    "BrainAge": "BrainAge",
    "BrainAgeR": "BrainAgeR",
    "DeepBrainNet": "DeepBrainNet",
    "Pyment": "Pyment",
    "BRAID_WM": "BRAID WM",
    "BRAID_GM": "BRAID GM",
}


def anova_effect_sizes_oneway(sub: pd.DataFrame) -> dict[str, float]:
    """Return eta-squared, omega-squared, and Cohen's f."""
    vals = [
        sub.loc[sub[GROUP_COL] == g, "BrainAgeGap"].dropna().to_numpy(float)
        for g in GROUP_ORDER
    ]
    vals = [v for v in vals if len(v) > 0]
    if len(vals) < 2:
        return {"eta2": np.nan, "omega2": np.nan, "f": np.nan}

    all_vals = np.concatenate(vals)
    grand = all_vals.mean()
    ss_between = sum(len(v) * (v.mean() - grand) ** 2 for v in vals)
    ss_within = sum(((v - v.mean()) ** 2).sum() for v in vals)
    ss_total = ss_between + ss_within
    k = len(vals)
    n = len(all_vals)
    ms_within = ss_within / (n - k)

    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    omega2 = (
        (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
        if ss_total + ms_within > 0
        else np.nan
    )
    f = np.sqrt(eta2 / (1 - eta2)) if 0 <= eta2 < 1 else np.nan
    return {"eta2": eta2, "omega2": omega2, "f": f}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_context("talk", font_scale=1.2)

    metric = "BAG_corr" if USE_CORRECTED else "BAG_raw"
    bag_cols = [f"{metric}_{m}" for m in MODEL_NAMES]

    df = pd.read_excel(INPUT_XLSX, sheet_name="Analysis_Data")
    missing = [c for c in [SUBJECT_COL, GROUP_COL, *bag_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_long = df.melt(
        id_vars=[SUBJECT_COL, GROUP_COL],
        value_vars=bag_cols,
        var_name="Model",
        value_name="BrainAgeGap",
    )
    df_long["Model"] = df_long["Model"].str.replace(f"{metric}_", "", regex=False)
    df_long[GROUP_COL] = pd.Categorical(
        df_long[GROUP_COL], categories=GROUP_ORDER, ordered=True
    )
    df_long = df_long.dropna(subset=["BrainAgeGap", SUBJECT_COL, GROUP_COL]).copy()

    full = smf.mixedlm(
        "BrainAgeGap ~ C(group) * C(Model)",
        data=df_long,
        groups=df_long[SUBJECT_COL],
    ).fit(reml=False)
    reduced = smf.mixedlm(
        "BrainAgeGap ~ C(group) + C(Model)",
        data=df_long,
        groups=df_long[SUBJECT_COL],
    ).fit(reml=False)

    lr = 2 * (full.llf - reduced.llf)
    df_diff = max(int(full.df_modelwc - reduced.df_modelwc), 1)
    p_lrt = chi2.sf(lr, df_diff)
    print(f"Group × Model LRT: LR={lr:.3f}, df={df_diff}, p={p_lrt:.6g}")

    rows = []
    raw_p = []

    for model in MODEL_NAMES:
        sub = df_long[df_long["Model"] == model].copy()
        groups = [
            sub.loc[sub[GROUP_COL] == g, "BrainAgeGap"].dropna().to_numpy(float)
            for g in GROUP_ORDER
        ]
        f_stat, p_val = f_oneway(*groups)
        effect = anova_effect_sizes_oneway(sub)
        raw_p.append(p_val)

        tk = pairwise_tukeyhsd(
            endog=sub["BrainAgeGap"],
            groups=sub[GROUP_COL],
            alpha=0.05,
        )
        print(f"\n{PRETTY_NAMES[model]} Tukey HSD")
        print(tk.summary())

        rows.append(
            {
                "model": PRETTY_NAMES[model],
                "F": f_stat,
                "p_raw": p_val,
                "eta2": effect["eta2"],
                "omega2": effect["omega2"],
                "cohen_f": effect["f"],
            }
        )

    reject, p_fdr, _, _ = multipletests(raw_p, method=MULTIPLICITY_METHOD)
    for row, pc, rj in zip(rows, p_fdr, reject):
        row["p_FDR"] = pc
        row["FDR_significant"] = bool(rj)

    summary = pd.DataFrame(rows)
    summary.insert(0, "metric", metric)
    summary.to_csv(OUTPUT_DIR / f"model_screening_{metric}.csv", index=False)
    print(summary.to_string(index=False))

    plt.figure(figsize=(18, 8))
    df_plot = df_long.copy()
    df_plot["Model"] = df_plot["Model"].map(PRETTY_NAMES)

    ax = sns.violinplot(
        data=df_plot,
        x="Model",
        y="BrainAgeGap",
        hue=GROUP_COL,
        order=[PRETTY_NAMES[m] for m in MODEL_NAMES],
        hue_order=GROUP_ORDER,
        inner="box",
        cut=0,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel(f"{metric} (years)")
    ax.set_title(
        "Age-bias-corrected BAG by model and group"
        if USE_CORRECTED
        else "Uncorrected BAG by model and group"
    )
    ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    out_png = OUTPUT_DIR / f"model_screening_{metric}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
