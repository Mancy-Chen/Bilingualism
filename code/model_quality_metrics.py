#!/usr/bin/env python3
"""Compute model-quality metrics for all six pretrained brain-age models."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

INPUT_XLSX = Path("input/brainpad_results_deidentified.xlsx")
OUTPUT_DIR = Path("output")
SHEET_NAME = "Analysis_Data"
AGE_COL = "Age"

MODEL_NAMES = [
    "BrainAge",
    "BrainAgeR",
    "DeepBrainNet",
    "Pyment",
    "BRAID_WM",
    "BRAID_GM",
]


def age_bag_slope(age: pd.Series, bag_uncorr: pd.Series) -> tuple[float, float, float]:
    """Return slope, standard error, and p-value for BAG_uncorr ~ Age."""
    dat = pd.DataFrame({"Age": age, "BAG_uncorr": bag_uncorr}).dropna()
    if len(dat) < 3:
        return np.nan, np.nan, np.nan

    x = sm.add_constant(dat["Age"].astype(float), has_constant="add")
    fit = sm.OLS(dat["BAG_uncorr"].astype(float), x).fit()
    return (
        float(fit.params["Age"]),
        float(fit.bse["Age"]),
        float(fit.pvalues["Age"]),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

    if AGE_COL not in df.columns:
        raise ValueError(f"Missing age column: {AGE_COL!r}")

    rows = []
    for model in MODEL_NAMES:
        pred_col = f"PredAge_{model}"
        bag_col = f"BAG_uncorr_{model}"
        missing = [c for c in [pred_col, bag_col] if c not in df.columns]
        if missing:
            raise ValueError(f"{model}: missing required columns: {missing}")

        dat = df[[AGE_COL, pred_col, bag_col]].apply(pd.to_numeric, errors="coerce")
        valid_pred = dat[[AGE_COL, pred_col]].dropna()
        errors = valid_pred[pred_col] - valid_pred[AGE_COL]

        slope, slope_se, slope_p = age_bag_slope(dat[AGE_COL], dat[bag_col])

        rows.append(
            {
                "model": model,
                "n_prediction": int(len(valid_pred)),
                "MAE_years": float(np.mean(np.abs(errors))),
                "RMSE_years": float(np.sqrt(np.mean(np.square(errors)))),
                "mean_prediction_error_years": float(np.mean(errors)),
                "age_BAG_uncorr_slope": slope,
                "age_BAG_uncorr_slope_SE": slope_se,
                "age_BAG_uncorr_slope_p": slope_p,
            }
        )

    summary = pd.DataFrame(rows)
    summary["abs_age_BAG_uncorr_slope"] = summary["age_BAG_uncorr_slope"].abs()
    summary.to_csv(OUTPUT_DIR / "model_quality_metrics.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'model_quality_metrics.csv'}")


if __name__ == "__main__":
    main()
