#!/usr/bin/env python3
"""
Effect-coded ROI group models for the revised manuscript.

Reproduces:
- average volume–BAG_corr association across the three groups
- omnibus volume_resid × group interaction tests
- descriptive within-group OLS slopes

ROI volumes are residualised for ICV, age, and sex before analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

GROUPS = ["bilinguals", "translators", "interpreters"]


def canonical_group(value: object) -> str:
    s = str(value).strip().lower()
    aliases = {
        "general_bilingual": "bilinguals",
        "general-bilingual": "bilinguals",
        "bilingual": "bilinguals",
        "bilinguals": "bilinguals",
        "translator": "translators",
        "translators": "translators",
        "interpreter": "interpreters",
        "interpreters": "interpreters",
    }
    return aliases.get(s, s)


def residualize_roi(sub: pd.DataFrame, covars: list[str]) -> pd.DataFrame:
    """Residualize ROI volume for ICV, age, and sex."""
    sub = sub.copy()
    y = pd.to_numeric(sub["volume_ml"], errors="coerce")

    parts = []
    for cov in covars:
        s = sub[cov]
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().sum() >= max(5, int(0.5 * len(s))):
            parts.append(pd.DataFrame({cov: numeric}, index=sub.index))
        else:
            parts.append(
                pd.get_dummies(
                    s.astype("category"),
                    prefix=cov,
                    drop_first=True,
                    dtype=float,
                )
            )

    X = pd.concat(parts, axis=1)
    X = sm.add_constant(X, has_constant="add").astype(float)

    valid = y.notna() & X.notna().all(axis=1)
    resid = pd.Series(np.nan, index=sub.index, dtype=float)

    if valid.sum() >= 8 and X.loc[valid].shape[1] < valid.sum():
        fit = sm.OLS(y.loc[valid].astype(float), X.loc[valid]).fit()
        resid.loc[valid] = fit.resid

    sub["volume_resid"] = resid
    return sub


def apply_fdr(
    df: pd.DataFrame,
    p_col: str,
    fdr_col: str,
    sig_col: str = "FDR_significant",
) -> pd.DataFrame:
    out = df.copy()
    out[fdr_col] = np.nan
    out[sig_col] = False

    pvals = pd.to_numeric(out[p_col], errors="coerce").to_numpy(float)
    valid = np.isfinite(pvals)
    if valid.any():
        reject, corrected, _, _ = multipletests(pvals[valid], method="fdr_bh")
        idx = out.index.to_numpy()[valid]
        out.loc[idx, fdr_col] = corrected
        out.loc[idx, sig_col] = reject
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--brain",
        default="input/brainpad_results_deidentified.xlsx",
    )
    parser.add_argument(
        "--roi",
        default="input/roi_volumes_deidentified.csv",
    )
    parser.add_argument(
        "--bag-col",
        default="BAG_corr_BrainAge",
    )
    parser.add_argument(
        "--output",
        default="output/roi_group_models",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    brain = pd.read_excel(args.brain, sheet_name="Analysis_Data")
    roi = pd.read_csv(args.roi)

    required_brain = {
        "MRI code",
        "group",
        "Age",
        "Gender",
        "ICV_ml",
        args.bag_col,
    }
    required_roi = {"subject_id", "label_id", "roi_name", "volume_ml"}

    missing_brain = required_brain - set(brain.columns)
    missing_roi = required_roi - set(roi.columns)
    if missing_brain:
        raise ValueError(f"Brain table missing columns: {sorted(missing_brain)}")
    if missing_roi:
        raise ValueError(f"ROI table missing columns: {sorted(missing_roi)}")

    brain = brain.copy()
    brain["subject_id"] = "sub-" + brain["MRI code"].astype(str).str.strip()
    brain["group_std"] = brain["group"].map(canonical_group)

    merged = roi.merge(
        brain[
            [
                "subject_id",
                "group_std",
                "Age",
                "Gender",
                "ICV_ml",
                args.bag_col,
            ]
        ],
        on="subject_id",
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("ROI/brain merge is empty. Check anonymous subject IDs.")

    merged = (
        merged.groupby("roi_name", group_keys=False)
        .apply(
            residualize_roi,
            covars=["ICV_ml", "Age", "Gender"],
        )
        .reset_index(drop=True)
    )

    main_rows = []
    interaction_rows = []
    slope_rows = []

    for roi_name, sub in merged.groupby("roi_name"):
        label_id = int(sub["label_id"].iloc[0])

        model_dat = sub[
            ["group_std", "volume_resid", args.bag_col]
        ].dropna().copy()

        if len(model_dat) >= 8 and model_dat["group_std"].nunique() == 3:
            model_dat["group_std"] = pd.Categorical(
                model_dat["group_std"],
                categories=GROUPS,
                ordered=True,
            )

            reduced = smf.ols(
                f'Q("{args.bag_col}") ~ volume_resid + C(group_std, Sum)',
                data=model_dat,
            ).fit()
            full = smf.ols(
                f'Q("{args.bag_col}") ~ volume_resid * C(group_std, Sum)',
                data=model_dat,
            ).fit()

            cmp = anova_lm(reduced, full)

            main_rows.append(
                {
                    "roi_name": roi_name,
                    "label_id": label_id,
                    "n": int(len(model_dat)),
                    "beta_volume_average": float(full.params["volume_resid"]),
                    "SE_volume_average": float(full.bse["volume_resid"]),
                    "t_volume_average": float(full.tvalues["volume_resid"]),
                    "p_volume_average_raw": float(
                        full.pvalues["volume_resid"]
                    ),
                }
            )

            interaction_rows.append(
                {
                    "roi_name": roi_name,
                    "label_id": label_id,
                    "n": int(len(model_dat)),
                    "F_interaction": float(cmp["F"].iloc[1]),
                    "df_num": int(cmp["df_diff"].iloc[1]),
                    "df_den": int(full.df_resid),
                    "p_interaction_raw": float(cmp["Pr(>F)"].iloc[1]),
                }
            )

        for group in GROUPS:
            gdat = sub.loc[
                sub["group_std"] == group,
                ["volume_resid", args.bag_col],
            ].dropna()

            if len(gdat) < 3 or gdat["volume_resid"].std() == 0:
                continue

            X = sm.add_constant(
                gdat["volume_resid"].astype(float),
                has_constant="add",
            )
            fit = sm.OLS(
                gdat[args.bag_col].astype(float),
                X,
            ).fit()

            slope_rows.append(
                {
                    "group": group,
                    "roi_name": roi_name,
                    "label_id": label_id,
                    "n": int(fit.nobs),
                    "slope_BAG_per_mL": float(fit.params["volume_resid"]),
                    "SE": float(fit.bse["volume_resid"]),
                    "t": float(fit.tvalues["volume_resid"]),
                    "p_raw": float(fit.pvalues["volume_resid"]),
                }
            )

    main_effects = pd.DataFrame(main_rows)
    interactions = pd.DataFrame(interaction_rows)
    slopes = pd.DataFrame(slope_rows)

    main_effects = apply_fdr(
        main_effects,
        p_col="p_volume_average_raw",
        fdr_col="p_volume_average_FDR",
    )
    interactions = apply_fdr(
        interactions,
        p_col="p_interaction_raw",
        fdr_col="p_interaction_FDR",
    )

    main_effects.to_csv(
        output_dir / "roi_main_effects.csv",
        index=False,
    )
    interactions.to_csv(
        output_dir / "roi_group_interactions.csv",
        index=False,
    )
    slopes.to_csv(
        output_dir / "roi_within_group_slopes.csv",
        index=False,
    )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
