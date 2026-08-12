#!/usr/bin/env python3
"""
Exploratory ROI-wise structure–BAG analysis for the selected BrainAge model.

Inputs
------
- input/brainpad_results_deidentified.xlsx
- input/roi_volumes_deidentified.csv
- optional FastSurfer DKT+ASEG label template for NIfTI heatmaps

The default outcome is ``BAG_corr_BrainAge``.
ROI volumes are residualised for ICV, age, and sex before group-wise
associations are estimated.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm, pearsonr, spearmanr
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

GROUPS = ["bilinguals", "translators", "interpreters"]
CORR_TYPES = ["pearson", "spearman"]


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
    """Residualize volume_ml for numeric/categorical covariates."""
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

    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=sub.index)
    X = sm.add_constant(X, has_constant="add").astype(float)

    valid = y.notna() & X.notna().all(axis=1)
    resid = pd.Series(np.nan, index=sub.index, dtype=float)

    if valid.sum() >= 8 and X.loc[valid].shape[1] < valid.sum():
        fit = sm.OLS(y.loc[valid].astype(float), X.loc[valid]).fit()
        resid.loc[valid] = fit.resid
    else:
        resid.loc[valid] = y.loc[valid]

    sub["volume_resid"] = resid
    return sub


def safe_corr(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
    min_n: int,
) -> tuple[int, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_n:
        return n, np.nan, np.nan

    xx = x[mask]
    yy = y[mask]
    if np.nanstd(xx) == 0 or np.nanstd(yy) == 0:
        return n, np.nan, np.nan

    if kind == "pearson":
        r, p = pearsonr(xx, yy)
    else:
        r, p = spearmanr(xx, yy)
    return n, float(r), float(p)


def fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    if not (np.isfinite(r1) and np.isfinite(r2)) or n1 <= 3 or n2 <= 3:
        return np.nan, np.nan
    r1 = np.clip(r1, -0.999999, 0.999999)
    r2 = np.clip(r2, -0.999999, 0.999999)
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(
        1 / (n1 - 3) + 1 / (n2 - 3)
    )
    p = 2 * norm.sf(abs(z))
    return float(z), float(p)


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
        "--template",
        default="input/aparc.DKTatlas+aseg.deep.withCC.mgz",
    )
    parser.add_argument("--output", default="output/roi_analysis")
    parser.add_argument("--bag-col", default="BAG_corr_BrainAge")
    parser.add_argument("--min-n", type=int, default=15)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    brain = pd.read_excel(args.brain, sheet_name="Analysis_Data")
    roi = pd.read_csv(args.roi)

    required_brain = {"MRI code", "group", "Age", "Gender", "ICV_ml", args.bag_col}
    missing_brain = required_brain - set(brain.columns)
    if missing_brain:
        raise ValueError(f"Brain table missing columns: {sorted(missing_brain)}")

    required_roi = {"subject_id", "label_id", "roi_name", "volume_ml"}
    missing_roi = required_roi - set(roi.columns)
    if missing_roi:
        raise ValueError(f"ROI table missing columns: {sorted(missing_roi)}")

    brain = brain.copy()
    brain["subject_id"] = "sub-" + brain["MRI code"].astype(str).str.strip()
    brain["group_std"] = brain["group"].map(canonical_group)

    merge_cols = [
        "subject_id",
        "group_std",
        "Age",
        "Gender",
        "ICV_ml",
        args.bag_col,
    ]
    merged = roi.merge(brain[merge_cols], on="subject_id", how="inner")
    if merged.empty:
        raise RuntimeError("ROI/brain merge is empty. Check anonymous subject IDs.")

    covars = ["ICV_ml", "Age", "Gender"]
    merged = (
        merged.groupby("roi_name", group_keys=False)
        .apply(residualize_roi, covars=covars)
        .reset_index(drop=True)
    )

    association_rows = []
    for group in GROUPS:
        gdat = merged[merged["group_std"] == group]
        for roi_name, sub in gdat.groupby("roi_name"):
            label_id = int(sub["label_id"].iloc[0])
            x = pd.to_numeric(sub["volume_resid"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(sub[args.bag_col], errors="coerce").to_numpy(float)

            for kind in CORR_TYPES:
                n, r, p = safe_corr(x, y, kind, args.min_n)
                association_rows.append(
                    {
                        "group": group,
                        "roi_name": roi_name,
                        "label_id": label_id,
                        "correlation": kind,
                        "n": n,
                        "r": r,
                        "p_raw": p,
                    }
                )

    assoc = pd.DataFrame(association_rows)
    assoc["p_FDR"] = np.nan
    assoc["FDR_significant"] = False

    for (_, _), idx in assoc.groupby(["group", "correlation"]).groups.items():
        pvals = assoc.loc[idx, "p_raw"].to_numpy(float)
        valid = np.isfinite(pvals)
        if valid.any():
            reject, pcorr, _, _ = multipletests(pvals[valid], method="fdr_bh")
            valid_idx = np.asarray(list(idx))[valid]
            assoc.loc[valid_idx, "p_FDR"] = pcorr
            assoc.loc[valid_idx, "FDR_significant"] = reject

    assoc.to_csv(output_dir / "roi_associations.csv", index=False)

    fisher_rows = []
    lookup = assoc.set_index(["group", "roi_name", "correlation"])

    for kind in CORR_TYPES:
        for g1, g2 in itertools.combinations(GROUPS, 2):
            common_rois = sorted(
                set(
                    assoc.loc[
                        (assoc.group == g1) & (assoc.correlation == kind),
                        "roi_name",
                    ]
                )
                & set(
                    assoc.loc[
                        (assoc.group == g2) & (assoc.correlation == kind),
                        "roi_name",
                    ]
                )
            )
            for roi_name in common_rois:
                a = lookup.loc[(g1, roi_name, kind)]
                b = lookup.loc[(g2, roi_name, kind)]
                z, p = fisher_z_test(
                    float(a["r"]), int(a["n"]), float(b["r"]), int(b["n"])
                )
                fisher_rows.append(
                    {
                        "correlation": kind,
                        "group1": g1,
                        "group2": g2,
                        "roi_name": roi_name,
                        "label_id": int(a["label_id"]),
                        "r1": float(a["r"]),
                        "n1": int(a["n"]),
                        "r2": float(b["r"]),
                        "n2": int(b["n"]),
                        "z": z,
                        "p_raw": p,
                    }
                )

    fisher = pd.DataFrame(fisher_rows)
    if not fisher.empty:
        fisher["p_FDR"] = np.nan
        fisher["FDR_significant"] = False
        for _, idx in fisher.groupby(["correlation", "group1", "group2"]).groups.items():
            pvals = fisher.loc[idx, "p_raw"].to_numpy(float)
            valid = np.isfinite(pvals)
            if valid.any():
                reject, pcorr, _, _ = multipletests(pvals[valid], method="fdr_bh")
                valid_idx = np.asarray(list(idx))[valid]
                fisher.loc[valid_idx, "p_FDR"] = pcorr
                fisher.loc[valid_idx, "FDR_significant"] = reject
        fisher.to_csv(output_dir / "roi_fisher_z.csv", index=False)

    interaction_rows = []
    for roi_name, sub in merged.groupby("roi_name"):
        sub = sub[["group_std", "volume_resid", args.bag_col]].dropna().copy()
        if len(sub) < args.min_n or sub["group_std"].nunique() < 2:
            continue

        sub["group_std"] = pd.Categorical(sub["group_std"], categories=GROUPS)
        try:
            reduced = smf.ols(
                f'Q("{args.bag_col}") ~ volume_resid + C(group_std)',
                data=sub,
            ).fit()
            full = smf.ols(
                f'Q("{args.bag_col}") ~ volume_resid * C(group_std)',
                data=sub,
            ).fit()
            cmp = anova_lm(reduced, full)
            p_interaction = float(cmp["Pr(>F)"].iloc[1])
            f_interaction = float(cmp["F"].iloc[1])

            interaction_rows.append(
                {
                    "roi_name": roi_name,
                    "label_id": int(
                        merged.loc[merged["roi_name"] == roi_name, "label_id"].iloc[0]
                    ),
                    "n": int(len(sub)),
                    "F_interaction": f_interaction,
                    "p_interaction_raw": p_interaction,
                }
            )
        except Exception:
            continue

    interactions = pd.DataFrame(interaction_rows)
    if not interactions.empty:
        reject, pcorr, _, _ = multipletests(
            interactions["p_interaction_raw"].to_numpy(float),
            method="fdr_bh",
        )
        interactions["p_interaction_FDR"] = pcorr
        interactions["FDR_significant"] = reject
        interactions.to_csv(output_dir / "roi_group_interactions.csv", index=False)

    template_path = Path(args.template)
    if template_path.exists():
        img = nib.load(str(template_path))
        labels = np.asarray(img.dataobj)

        for group in GROUPS:
            for kind in CORR_TYPES:
                sub = assoc[
                    (assoc["group"] == group)
                    & (assoc["correlation"] == kind)
                    & (assoc["FDR_significant"])
                ]
                out = np.zeros(labels.shape, dtype=np.float32)
                for _, row in sub.iterrows():
                    out[labels == int(row["label_id"])] = float(row["r"])

                nii = nib.Nifti1Image(out, img.affine, img.header)
                fname = f"{group}_{kind}_{args.bag_col}_FDR_heatmap.nii.gz"
                nib.save(nii, output_dir / fname)

    print(f"Saved ROI analysis outputs to: {output_dir}")


if __name__ == "__main__":
    main()
