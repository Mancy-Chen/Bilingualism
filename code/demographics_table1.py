#!/usr/bin/env python3
"""Reproduce the demographic summary and group-comparison tests for Table 1.

Expected input
--------------
input/brainpad_results_deidentified.xlsx, sheet ``Analysis_Data``.

The script summarises continuous variables as mean ± SD, sex as n (%), and
performs the same overall group comparisons described in the manuscript:

- one-way ANOVA for continuous participant characteristics
- chi-square test for sex distribution

The university-degree/current-enrolment row is included when a suitable
education indicator is present in the public analysis table. If the redundant
indicator is not stored, the manuscript's design-based 100% row can be added
explicitly with ``--include-design-based-education``; this option is labelled
as design-based in the output rather than inferred from participant data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway

GROUP_ORDER = ["bilinguals", "translators", "interpreters"]
GROUP_LABELS = {
    "bilinguals": "Bilinguals",
    "translators": "Translators",
    "interpreters": "Interpreters",
}

CONTINUOUS_VARIABLES = [
    ("Age, years", ["Age", "age"]),
    ("FSIQ", ["FSIQ", "fsiq"]),
    (
        "Age of L2 acquisition, years",
        ["AoA", "Age_of_L2_acquisition", "Age of L2 acquisition", "age_l2"],
    ),
    ("LexTALE score", ["LexTale", "LexTALE", "lextale", "LexTALE_score"]),
    ("ICV, mL", ["ICV_ml", "ICV", "icv_ml"]),
]

EDUCATION_CANDIDATES = [
    "University_degree_or_current_enrolment",
    "university_degree_or_current_enrolment",
    "University_degree_current_university_enrolment",
    "university_degree_current_university_enrolment",
    "University_degree_current_enrolment",
    "university_degree_current_enrolment",
    "University_education",
    "university_education",
    "University_degree",
    "university_degree",
]


def canonical_group(value: object) -> str:
    s = str(value).strip().lower()
    aliases = {
        "bilingual": "bilinguals",
        "bilinguals": "bilinguals",
        "general_bilingual": "bilinguals",
        "general-bilingual": "bilinguals",
        "translator": "translators",
        "translators": "translators",
        "interpreter": "interpreters",
        "interpreters": "interpreters",
    }
    return aliases.get(s, s)


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return str(lower_map[candidate.lower()])
    return None


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".lstrip("0")


def mean_sd(series: pd.Series) -> str:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return "—"
    return f"{x.mean():.1f} ± {x.std(ddof=1):.1f}"


def count_percent(mask: pd.Series, denominator: int) -> str:
    n = int(mask.sum())
    pct = 100 * n / denominator if denominator else np.nan
    return f"{n} ({pct:.1f}%)" if np.isfinite(pct) else "—"


def truthy(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(False, index=series.index)
    numeric_mask = numeric.notna()
    out.loc[numeric_mask] = numeric.loc[numeric_mask] > 0
    text = series.astype(str).str.strip().str.lower()
    out.loc[~numeric_mask] = text.loc[~numeric_mask].isin(
        {"yes", "y", "true", "1", "degree", "enrolled", "current", "university"}
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="input/brainpad_results_deidentified.xlsx",
        help="Deidentified subject-level workbook.",
    )
    parser.add_argument("--sheet", default="Analysis_Data")
    parser.add_argument("--output", default="output/table1_demographics.csv")
    parser.add_argument(
        "--education-col",
        default="",
        help="Optional explicit university degree/current enrolment indicator column.",
    )
    parser.add_argument(
        "--include-design-based-education",
        action="store_true",
        help=(
            "If no education indicator is stored, include the manuscript's 100% "
            "university-degree/current-enrolment row based on the study inclusion criterion."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = pd.read_excel(input_path, sheet_name=sheet)

    group_col = find_column(df, ["group", "Group"])
    sex_col = find_column(df, ["Gender", "Sex", "gender", "sex"])
    if group_col is None:
        raise ValueError("Could not find the group column.")
    if sex_col is None:
        raise ValueError("Could not find the sex/gender column.")

    df = df.copy()
    df["_group"] = df[group_col].map(canonical_group)
    unexpected = sorted(set(df["_group"].dropna()) - set(GROUP_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected group labels: {unexpected}")

    group_n = {g: int((df["_group"] == g).sum()) for g in GROUP_ORDER}
    total_n = int(df["_group"].isin(GROUP_ORDER).sum())

    col_names = {
        g: f"{GROUP_LABELS[g]} (n = {group_n[g]})" for g in GROUP_ORDER
    }
    total_col = f"Total (n = {total_n})"

    rows: list[dict[str, object]] = []

    # Continuous variables.
    for label, candidates in CONTINUOUS_VARIABLES:
        col = find_column(df, candidates)
        if col is None:
            print(f"Warning: skipped {label!r}; no matching column found.")
            continue

        groups_numeric = []
        row: dict[str, object] = {"Characteristic": label}
        for g in GROUP_ORDER:
            values = pd.to_numeric(
                df.loc[df["_group"] == g, col], errors="coerce"
            ).dropna()
            row[col_names[g]] = mean_sd(values)
            groups_numeric.append(values.to_numpy(float))

        # Match the manuscript layout, where total ICV is not displayed.
        row[total_col] = "—" if label == "ICV, mL" else mean_sd(df[col])
        if all(len(x) >= 2 for x in groups_numeric):
            _, p = f_oneway(*groups_numeric)
        else:
            p = np.nan
        row["p"] = format_p(float(p))
        rows.append(row)

    # Sex distribution: one overall chi-square p-value, not one p-value per level.
    sex_text = df[sex_col].astype(str).str.strip().str.lower()
    female_mask = sex_text.isin({"female", "f", "woman", "women"})
    male_mask = sex_text.isin({"male", "m", "man", "men"})

    contingency = []
    for g in GROUP_ORDER:
        idx = df["_group"] == g
        contingency.append([int((idx & female_mask).sum()), int((idx & male_mask).sum())])
    _, sex_p, _, _ = chi2_contingency(np.asarray(contingency))

    sex_header = {"Characteristic": "Sex, n (%)", total_col: "", "p": format_p(float(sex_p))}
    for g in GROUP_ORDER:
        sex_header[col_names[g]] = ""
    rows.append(sex_header)

    for level, mask in [("    Female", female_mask), ("    Male", male_mask)]:
        row = {"Characteristic": level, "p": "—"}
        for g in GROUP_ORDER:
            idx = df["_group"] == g
            denom = int(idx.sum())
            row[col_names[g]] = count_percent(idx & mask, denom)
        row[total_col] = count_percent(mask & df["_group"].isin(GROUP_ORDER), total_n)
        rows.append(row)

    # Education row when a participant-level indicator is available.
    education_col = args.education_col or find_column(df, EDUCATION_CANDIDATES)
    if education_col:
        if education_col not in df.columns:
            raise ValueError(f"Education column not found: {education_col!r}")
        edu = truthy(df[education_col])
        row = {
            "Characteristic": "University degree/current university enrolment",
            "p": "—",
        }
        for g in GROUP_ORDER:
            idx = df["_group"] == g
            row[col_names[g]] = count_percent(idx & edu, int(idx.sum()))
        row[total_col] = count_percent(edu & df["_group"].isin(GROUP_ORDER), total_n)
        rows.append(row)
    elif args.include_design_based_education:
        row = {
            "Characteristic": "University degree/current university enrolment",
            "p": "—",
        }
        for g in GROUP_ORDER:
            n = group_n[g]
            row[col_names[g]] = f"{n} (100%)"
        row[total_col] = f"{total_n} (100%)"
        rows.append(row)
        print(
            "Note: university education row was added from the predefined study "
            "inclusion criterion because no participant-level indicator was found."
        )
    else:
        print(
            "Note: no education indicator found. Use --education-col COLUMN or "
            "--include-design-based-education to include that Table 1 row."
        )

    table = pd.DataFrame(rows)
    ordered_cols = [
        "Characteristic",
        *[col_names[g] for g in GROUP_ORDER],
        total_col,
        "p",
    ]
    table = table.reindex(columns=ordered_cols)
    table.to_csv(output_path, index=False)

    print(table.to_string(index=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
