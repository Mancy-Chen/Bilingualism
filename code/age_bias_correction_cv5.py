#!/usr/bin/env python3
"""Five-fold cross-validated linear age-bias correction for brain age gap (BAG).

Final notation
--------------
PredAge_<Model>    : predicted brain age in years
BAG_uncorr_<Model> : uncorrected brain-age gap = predicted brain age - chronological age
BAG_corr_<Model>   : five-fold cross-validated age-bias-corrected BAG

The deposited analysis dataset already contains the final BAG_corr values used
in the manuscript. This script documents and regenerates the correction
procedure from BAG_uncorr using the project settings:

    n_splits=5
    shuffle=True
    random_state=42

The age-bias term is estimated within each training fold and applied only to the
held-out fold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold

UNCORR_PREFIX = "BAG_uncorr_"
CORR_PREFIX = "BAG_corr_"


def find_uncorrected_bag_columns(
    df: pd.DataFrame,
    prefix: str = UNCORR_PREFIX,
) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(
            f"No uncorrected BAG columns found with prefix {prefix!r}. "
            "Expected columns such as 'BAG_uncorr_BrainAge'."
        )
    return cols


def fit_age_bias(y: np.ndarray, age: np.ndarray) -> tuple[float, float]:
    """Fit BAG_uncorr = alpha + beta * Age on non-missing training observations."""
    valid = np.isfinite(y) & np.isfinite(age)
    if valid.sum() < 3:
        raise RuntimeError("Too few non-missing observations to fit age bias.")
    X = sm.add_constant(age[valid], has_constant="add")
    fit = sm.OLS(y[valid], X).fit()
    return float(fit.params[0]), float(fit.params[1])


def generate_folds(
    n_rows: int,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Generate reproducible KFold assignments numbered 1..n_splits."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_id = np.full(n_rows, -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(kf.split(np.arange(n_rows)), start=1):
        fold_id[test_idx] = fold_idx
    return fold_id


def correct_bag_with_folds(
    df: pd.DataFrame,
    uncorr_bag_cols: list[str],
    age_col: str,
    fold_id: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate BAG_corr for each model using held-out-fold age-bias estimates."""
    age = pd.to_numeric(df[age_col], errors="coerce").to_numpy(dtype=float)
    outputs: dict[str, np.ndarray] = {}
    parameter_rows: list[dict] = []

    for uncorr_col in uncorr_bag_cols:
        model = uncorr_col[len(UNCORR_PREFIX):]
        corr_col = f"{CORR_PREFIX}{model}"
        y = pd.to_numeric(df[uncorr_col], errors="coerce").to_numpy(dtype=float)
        corr_out = np.full(len(df), np.nan, dtype=float)

        for fold in sorted(np.unique(fold_id)):
            train_idx = np.flatnonzero(fold_id != fold)
            test_idx = np.flatnonzero(fold_id == fold)

            alpha, beta = fit_age_bias(y[train_idx], age[train_idx])
            expected_age_bias = alpha + beta * age[test_idx]
            corr_out[test_idx] = y[test_idx] - expected_age_bias

            parameter_rows.append(
                {
                    "model": model,
                    "fold": int(fold),
                    "alpha": alpha,
                    "beta_age": beta,
                    "n_train_nonmissing": int(
                        np.sum(np.isfinite(y[train_idx]) & np.isfinite(age[train_idx]))
                    ),
                    "n_test_nonmissing": int(
                        np.sum(np.isfinite(y[test_idx]) & np.isfinite(age[test_idx]))
                    ),
                }
            )

        outputs[corr_col] = corr_out

    return pd.DataFrame(outputs, index=df.index), pd.DataFrame(parameter_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Five-fold cross-validated linear age-bias correction of BAG_uncorr."
    )
    parser.add_argument(
        "--input",
        default="input/brainpad_results_deidentified.xlsx",
        help="Input .xlsx file.",
    )
    parser.add_argument(
        "--output",
        default="output/brainpad_results_bias_corrected.xlsx",
        help="Output .xlsx file.",
    )
    parser.add_argument("--sheet", default="Analysis_Data")
    parser.add_argument("--age-col", default="Age")
    parser.add_argument("--uncorr-prefix", default=UNCORR_PREFIX)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = pd.read_excel(input_path, sheet_name=sheet)

    if args.age_col not in df.columns:
        raise ValueError(f"Missing age column: {args.age_col!r}")

    uncorr_bag_cols = find_uncorrected_bag_columns(df, prefix=args.uncorr_prefix)
    fold_id = generate_folds(
        n_rows=len(df),
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    corrected_df, params_df = correct_bag_with_folds(
        df=df,
        uncorr_bag_cols=uncorr_bag_cols,
        age_col=args.age_col,
        fold_id=fold_id,
    )

    out_df = df.copy()
    for col in corrected_df.columns:
        out_df[col] = corrected_df[col]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Analysis_Data", index=False)
        params_df.to_excel(writer, sheet_name="cv_parameters", index=False)

    print(f"Input:  {input_path}")
    print(f"Rows:   {len(df)}")
    print(f"Models: {len(uncorr_bag_cols)}")
    print(f"Folds:  {args.n_splits}")
    print(f"Saved:  {output_path}")


if __name__ == "__main__":
    main()
