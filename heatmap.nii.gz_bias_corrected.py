###############################################################################################################################
# New heatmap.nii.gz for bias adjusted Brain age gap and corrected ROI
# -*- coding: utf-8 -*-
"""
ROI–BrainPAD correlation maps (Pearson & Spearman), with residualization, FDR, and between-group tests.
Saves separate r heatmaps for bilinguals / translators / interpreters.

Requirements:
  pip install pandas numpy nibabel scipy statsmodels
"""

import os
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, norm
from statsmodels.stats.multitest import multipletests

# ---------------------- Paths & constants ----------------------
BRAINPAD_XLSX = Path("/data/projects/CSC/code/Bilingualism/09_output_compare/brainpad_results_radiomics.xlsx")
ROI_LONG_CSV  = Path("/data/projects/CSC/code/Bilingualism/10_Fastsurfer/roi_volumes/DKTatlas_aseg_deep_withCC_long.csv")
TEMPLATE_SEG  = Path("/data/projects/CSC/code/Bilingualism/10_Fastsurfer/my_fastsurfer_analysis_n4/sub-3777B/sub-3777B/mri/aparc.DKTatlas+aseg.deep.withCC.mgz")
OUTPUT_DIR    = Path("/data/projects/CSC/code/Bilingualism/09_output_compare/roi_heatmaps_cv5_resid_FDR_n4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use only CV5 bias-corrected deltas
DELTA_PREFIX = "delta_cv5_Predicted_age_non_BC_"

# Minimum paired samples to compute correlation
MIN_N = 15

# Covariates to residualize ROI volumes; will drop any that are not present
COVAR_PREFS = ["VoxelVolume_mL", "Age", "Gender"]

# Correlation types to compute
CORR_TYPES = ["pearson", "spearman"]  # both

# Significance threshold for masking (FDR q-value)
ALPHA = 0.05

# Target groups to save heatmaps for (canonical names)
TARGET_GROUPS = ["bilinguals", "translators", "interpreters"]

# Map raw group labels to canonical names
GROUP_CANONICAL_MAP = {
    "general_bilingual": "bilinguals",
    "general-bilingual": "bilinguals",
    "bilingual": "bilinguals",
    "bilinguals": "bilinguals",
    "translator": "translators",
    "translators": "translators",
    "interpreter": "interpreters",
    "interpreters": "interpreters",
}

def canon_group(g):
    if pd.isna(g):
        return np.nan
    s = str(g).strip().lower()
    return GROUP_CANONICAL_MAP.get(s, s)  # fallback: lower-cased original

# ---------------------- Load data ----------------------
print("Loading brain-age deltas and ROI volumes...")
brain = pd.read_excel(BRAINPAD_XLSX)

# Normalize IDs: brain has "MRI code" (plain 'xxxxB'); ROI uses subject_id like 'sub-xxxxB'
if "MRI code" not in brain.columns:
    raise ValueError("Expected column 'MRI code' in brainpad_results.xlsx")
brain = brain.rename(columns={"MRI code": "subject_plain"})
brain["subject_id"] = "sub-" + brain["subject_plain"].astype(str).str.strip()

# Ensure group column exists and canonicalize
if "group" not in brain.columns:
    raise ValueError("Column 'group' not found in brainpad_results.xlsx")
brain["group_std"] = brain["group"].apply(canon_group)

# Keep only the delta_cv5_* models
delta_cols = [c for c in brain.columns if c.startswith(DELTA_PREFIX)]
if not delta_cols:
    raise ValueError(f"No columns found starting with '{DELTA_PREFIX}'. "
                     f"Please ensure your Excel has delta_cv5_Predicted_age_non_BC_* columns.")

# Load ROI long table
roi = pd.read_csv(ROI_LONG_CSV)

# Ensure required ROI fields
required_roi_cols = {"subject_id", "roi_name", "label_id", "volume_ml"}
missing = required_roi_cols - set(roi.columns)
if missing:
    raise ValueError(f"ROI table missing columns: {missing}")

# ---------------------- Merge brainpad covariates into ROI table ----------------------
covars_available = [c for c in COVAR_PREFS if c in brain.columns]
print(f"Residualization covariates available: {covars_available if covars_available else 'None'}")

merge_cols = ["subject_id", "group_std"] + covars_available + delta_cols
brain_for_merge = brain[merge_cols].copy()
merged = roi.merge(brain_for_merge, on="subject_id", how="inner")

if merged.empty:
    raise RuntimeError("Merged table is empty. Check subject_id harmonization and input tables.")

# ---------------------- Residualize ROI volumes ----------------------
# ---------- robust helpers (drop-in replacement) ----------
def _coerce_numeric_inplace(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def build_design_matrix(df_sub: pd.DataFrame, covar_names):
    """Intercept + numeric covariates (coerced) + one-hot categoricals; drop all-NaN/constant; float64."""
    X_parts = []

    for c in covar_names:
        if c not in df_sub.columns:
            continue
        s = df_sub[c]

        # If it looks numeric or mixed, try coercion first
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().sum() >= max(5, int(0.2 * len(s))):  # enough numerics → treat as numeric
            X_parts.append(pd.DataFrame({c: s_num}))
        else:
            # treat as categorical
            dummies = pd.get_dummies(s.astype("category"), prefix=c, drop_first=True)
            X_parts.append(dummies)

    X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=df_sub.index)

    # Drop all-NaN columns
    if not X.empty:
        X = X.loc[:, X.notna().any(axis=0)]

        # Drop constant columns (zero variance)
        nunique = X.nunique(dropna=True)
        X = X.loc[:, nunique > 1]

    # Add intercept
    X = sm.add_constant(X, has_constant="add")

    # Ensure float64 design
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)

    return X

def residualize_volume_per_roi(df_roi: pd.DataFrame, covar_names, min_n=8):
    """
    Residualize volume_ml ~ covariates (robust).
    Falls back to raw volume if not possible.
    """
    df_roi = df_roi.copy()

    # Coerce target and obvious numeric covariates
    _coerce_numeric_inplace(df_roi, ["volume_ml", "VoxelVolume_mL", "Age"])

    # Build design
    X = build_design_matrix(df_roi, covar_names)
    y = pd.to_numeric(df_roi["volume_ml"], errors="coerce").astype(float)

    # Valid rows: no NaN in X or y
    valid = X.notna().all(axis=1) & y.notna()
    n_valid = int(valid.sum())

    # Guard rails
    if n_valid < min_n or X.shape[1] >= n_valid:
        # Not enough data or overparameterized → fallback to raw volume
        df_roi["volume_resid"] = df_roi["volume_ml"]
        df_roi["resid_model_used"] = False
        return df_roi

    # Drop columns that became all-NaN after valid mask
    Xv = X.loc[valid]
    yv = y.loc[valid]

    # If any column is constant on valid rows, drop it
    nunique_v = Xv.nunique(dropna=True)
    const_cols = nunique_v[nunique_v <= 1].index.tolist()
    if const_cols:
        Xv = Xv.drop(columns=const_cols)
        # need intercept; add if we dropped it accidentally
        if "const" not in Xv.columns:
            Xv = sm.add_constant(Xv, has_constant="add")

    # Final check on rank / dims
    if Xv.shape[1] >= len(yv):
        df_roi["volume_resid"] = df_roi["volume_ml"]
        df_roi["resid_model_used"] = False
        return df_roi

    try:
        fit = sm.OLS(yv.values.astype(float), Xv.values.astype(float)).fit()
        yhat = fit.predict(Xv.values.astype(float))
        resid_full = pd.Series(index=df_roi.index, dtype=float)
        resid_full.loc[valid] = (yv.values - yhat)
        resid_full.loc[~valid] = np.nan
        df_roi["volume_resid"] = resid_full
        df_roi["resid_model_used"] = True
        return df_roi
    except Exception as e:
        # As a last resort, fallback
        df_roi["volume_resid"] = df_roi["volume_ml"]
        df_roi["resid_model_used"] = False
        return df_roi

print("Residualizing ROI volumes for covariates (robust)...")
merged = merged.groupby("roi_name", group_keys=False).apply(
    residualize_volume_per_roi, covar_names=covars_available, min_n=8
)


VOL_COL = "volume_resid"
if VOL_COL not in merged.columns:
    raise RuntimeError("Residualization failed to create 'volume_resid'.")
######slope starts here#######
# ---------------------- Safe correlation helpers ----------------------
def safe_corr(x, y, kind="pearson", min_n=MIN_N):
    mask = ~np.isnan(x) & ~np.isnan(y)
    n = int(mask.sum())
    if n < min_n:
        return np.nan, np.nan, n
    xx, yy = x[mask], y[mask]
    if kind == "pearson":
        r, p = pearsonr(xx, yy)
    elif kind == "spearman":
        r, p = spearmanr(xx, yy)
    else:
        raise ValueError(f"Unknown correlation kind: {kind}")
    return float(r), float(p), n

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def compare_r_fisher(r1, n1, r2, n2, min_n=MIN_N):
    if any([np.isnan(r1), np.isnan(r2)]) or min(n1, n2) < min_n or (n1 <= 3 or n2 <= 3):
        return np.nan, np.nan
    z1, z2 = fisher_z(r1), fisher_z(r2)
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)

# ---------------------- 1) Correlations by group (Pearson & Spearman) ----------------------
print("Computing correlations by group (Pearson & Spearman)...")

rows = []
# Use canonical group names
merged["group_std"] = merged["group_std"].apply(canon_group)
groups_all = sorted(merged["group_std"].dropna().astype(str).unique())

for corr_type in CORR_TYPES:
    for roi_name, grp_roi in merged.groupby("roi_name"):
        vol = grp_roi[VOL_COL].values
        # overall (ALL)
        for method in delta_cols:
            r_all, p_all, n_all = safe_corr(vol, grp_roi[method].values, kind=corr_type, min_n=MIN_N)
            rows.append({"corr": corr_type, "roi_name": roi_name, "method": method,
                         "group": "ALL", "n": n_all, "r": r_all, "p": p_all})
        # per canonical group
        for g, sub in grp_roi.groupby("group_std"):
            g = str(g)
            vol_g = sub[VOL_COL].values
            for method in delta_cols:
                r, p, n = safe_corr(vol_g, sub[method].values, kind=corr_type, min_n=MIN_N)
                rows.append({"corr": corr_type, "roi_name": roi_name, "method": method,
                             "group": g, "n": n, "r": r, "p": p})

df_corr = pd.DataFrame(rows).sort_values(["corr", "roi_name", "method", "group"]).reset_index(drop=True)

# ---------------------- 1b) FDR (BH) for within-group correlations ----------------------
print("Applying FDR to within-group correlations...")
df_corr["p_fdr"] = np.nan
for corr_type in CORR_TYPES:
    sub_ct = df_corr[df_corr["corr"] == corr_type]
    for (m, g), idx in sub_ct.groupby(["method", "group"]).groups.items():
        if g == "ALL":
            continue  # don't FDR-adjust the combined row with per-ROI sets? (optional)
        idx = list(idx)
        pvals = df_corr.loc[idx, "p"].values
        mask = np.isfinite(pvals)
        if mask.sum() > 0:
            _, p_adj, _, _ = multipletests(pvals[mask], alpha=ALPHA, method="fdr_bh")
            df_corr.loc[np.array(idx)[mask], "p_fdr"] = p_adj

corr_csv = OUTPUT_DIR / "roi_volume_vs_delta_corr_by_group_residICV_cv5.csv"
df_corr.to_csv(corr_csv, index=False)
print(f"Saved correlations (with FDR) -> {corr_csv}")

# ---------------------- 2) Between-group tests (Fisher r-to-z) with FDR ----------------------
print("Between-group Fisher r-to-z tests...")
pair_rows = []
# Only test pairs among our three canonical groups if they exist
present_targets = [g for g in TARGET_GROUPS if g in groups_all]

for corr_type in CORR_TYPES:
    sub_ct = df_corr[df_corr["corr"] == corr_type]
    for roi_name in sub_ct["roi_name"].unique():
        for method in delta_cols:
            sub = sub_ct[(sub_ct["roi_name"] == roi_name) &
                         (sub_ct["method"] == method) &
                         (sub_ct["group"] != "ALL")]
            stats_map = {row["group"]: (row["r"], int(row["n"])) for _, row in sub.iterrows()}
            for g1, g2 in itertools.combinations(present_targets, 2):
                r1, n1 = stats_map.get(str(g1), (np.nan, 0))
                r2, n2 = stats_map.get(str(g2), (np.nan, 0))
                z, p = compare_r_fisher(r1, n1, r2, n2, min_n=MIN_N)
                pair_rows.append({
                    "corr": corr_type,
                    "roi_name": roi_name,
                    "method": method,
                    "group1": str(g1), "group2": str(g2),
                    "r1": r1, "n1": n1, "r2": r2, "n2": n2,
                    "z": z, "p": p
                })

df_diff = pd.DataFrame(pair_rows).sort_values(["corr","roi_name","method","group1","group2"]).reset_index(drop=True)

# FDR over between-group tests (per method & corr_type)
print("Applying FDR to between-group tests...")
df_diff["p_fdr"] = np.nan
for corr_type in CORR_TYPES:
    sub = df_diff[df_diff["corr"] == corr_type]
    for m, idx in sub.groupby("method").groups.items():  # FDR within each method
        idx = list(idx)
        pvals = df_diff.loc[idx, "p"].values
        mask = np.isfinite(pvals)
        if mask.sum() > 0:
            _, p_adj, _, _ = multipletests(pvals[mask], alpha=ALPHA, method="fdr_bh")
            df_diff.loc[np.array(idx)[mask], "p_fdr"] = p_adj

diff_csv = OUTPUT_DIR / "roi_volume_vs_delta_between_group_tests_residICV_cv5.csv"
df_diff.to_csv(diff_csv, index=False)
print(f"Saved between-group tests (with FDR) -> {diff_csv}")

# ---------------------- 3) Heatmaps (r and FDR-masked r) for target groups ----------------------
print("Writing NIfTI heatmaps from ROI r-values for bilinguals/translators/interpreters...")
seg_img = nib.load(str(TEMPLATE_SEG))
seg_data = seg_img.get_fdata().astype(int)

# Map roi_name -> label_id (assume unique label per roi_name)
roi_map_df = roi.drop_duplicates(subset=["roi_name"])[["roi_name", "label_id"]]
roi_to_label = dict(zip(roi_map_df["roi_name"], roi_map_df["label_id"]))

def write_roi_map(value_map, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vol = np.zeros_like(seg_data, dtype=np.float32)
    for roi_name, val in value_map.items():
        lab = roi_to_label.get(roi_name, None)
        if lab is None:
            continue
        if np.isnan(val):
            continue
        vol[seg_data == int(lab)] = float(val)
    nib.save(nib.Nifti1Image(vol, affine=seg_img.affine, header=seg_img.header), str(out_path))

# Only for target groups; each group gets its own subfolder
for corr_type in CORR_TYPES:
    for method in delta_cols:
        for g in TARGET_GROUPS:
            if g not in groups_all:
                continue
            sub = df_corr[(df_corr["corr"] == corr_type) &
                          (df_corr["method"] == method) &
                          (df_corr["group"] == g)]
            r_map  = dict(zip(sub["roi_name"], sub["r"]))
            q_map  = dict(zip(sub["roi_name"], sub["p_fdr"]))  # FDR-adjusted p within that group

            group_dir = OUTPUT_DIR / g
            group_dir.mkdir(exist_ok=True, parents=True)

            # Raw r
            fname_r = f"heatmap_{corr_type}_r_{method}_group-{g}.nii.gz"
            write_roi_map(r_map, group_dir / fname_r)
            print("Saved", group_dir / fname_r)

            # r masked by FDR q < 0.05
            masked = {roi: (r if (q is not None and np.isfinite(q) and q < ALPHA) else 0.0)
                      for roi, r in r_map.items()
                      for q in [q_map.get(roi, np.nan)]}
            fname_mask = f"heatmap_{corr_type}_r_if_q_lt_0.05_{method}_group-{g}.nii.gz"
            write_roi_map(masked, group_dir / fname_mask)
            print("Saved", group_dir / fname_mask)

print("✅ All done.")
print(f"Outputs in: {OUTPUT_DIR} (and subfolders for each target group)")
##################################################
# Slope
# ====================== NEW: Slope-based analyses & heatmaps ======================
from scipy.stats import t as tdist  # for p-values in simple regression
OUTPUT_DIR_BETA = Path("/data/projects/CSC/code/Bilingualism/09_output_compare/roi_heatmaps_cv5_resid_FDR_beta_new_n4")
OUTPUT_DIR_BETA.mkdir(parents=True, exist_ok=True)
print("Computing per-group slopes (beta) of CV5 ΔBAG on volume_resid (ΔBAG per mL)...")

def safe_slope_ols(x, y, min_n=MIN_N):
    """
    Simple OLS for y ~ 1 + x.
    Returns (beta1, p, n). If not enough data or zero variance in x, returns (nan, nan, n).
    """
    x = pd.to_numeric(x, errors="coerce").values.astype(float)
    y = pd.to_numeric(y, errors="coerce").values.astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < max(min_n, 3):
        return np.nan, np.nan, n
    xx, yy = x[m], y[m]
    xbar = xx.mean()
    ybar = yy.mean()
    Sxx = np.sum((xx - xbar)**2)
    Sxy = np.sum((xx - xbar)*(yy - ybar))
    if Sxx <= 0:
        return np.nan, np.nan, n
    beta1 = Sxy / Sxx
    beta0 = ybar - beta1 * xbar
    # residual variance
    yhat = beta0 + beta1 * xx
    rss = np.sum((yy - yhat)**2)
    df  = n - 2
    if df <= 0:
        return np.nan, np.nan, n
    s2 = rss / df
    se_beta1 = np.sqrt(s2 / Sxx)
    if se_beta1 == 0 or not np.isfinite(se_beta1):
        return np.nan, np.nan, n
    tval = beta1 / se_beta1
    p = 2 * (1 - tdist.cdf(abs(tval), df=df))
    return float(beta1), float(p), n
# 1) Slopes by group (per ROI x delta method), with FDR like your correlation block
rows_beta = []
# Canonicalize groups (again, to be safe)
merged["group_std"] = merged["group_std"].apply(canon_group)
groups_all = sorted(merged["group_std"].dropna().astype(str).unique())
for roi_name, grp_roi in merged.groupby("roi_name"):
    vol = grp_roi[VOL_COL]
    # ALL
    for method in delta_cols:
        b, p, n = safe_slope_ols(vol, grp_roi[method], min_n=MIN_N)
        rows_beta.append({"roi_name": roi_name, "method": method, "group": "ALL",
                          "n": n, "beta": b, "p": p})
    # per canonical group
    for g, sub in grp_roi.groupby("group_std"):
        g = str(g)
        vol_g = sub[VOL_COL]
        for method in delta_cols:
            b, p, n = safe_slope_ols(vol_g, sub[method], min_n=MIN_N)
            rows_beta.append({"roi_name": roi_name, "method": method, "group": g,
                              "n": n, "beta": b, "p": p})
df_beta = pd.DataFrame(rows_beta).sort_values(["roi_name","method","group"]).reset_index(drop=True)
# FDR within-group (skip "ALL")
print("Applying FDR to per-group slope p-values...")
df_beta["p_fdr"] = np.nan
for (m, g), idx in df_beta.groupby(["method", "group"]).groups.items():
    if g == "ALL":
        continue
    idx = list(idx)
    pvals = df_beta.loc[idx, "p"].values
    mask = np.isfinite(pvals)
    if mask.sum() > 0:
        _, p_adj, _, _ = multipletests(pvals[mask], alpha=ALPHA, method="fdr_bh")
        df_beta.loc[np.array(idx)[mask], "p_fdr"] = p_adj
beta_csv = OUTPUT_DIR_BETA / "roi_volume_vs_delta_SLOPE_by_group_residICV_cv5.csv"
df_beta.to_csv(beta_csv, index=False)
print(f"Saved slopes (with FDR) -> {beta_csv}")
# 2) Between-group slope tests (ANCOVA-style interaction: volume_resid ~ 1 + x + G + x:G)
print("Between-group slope (beta) tests via interaction...")
pair_rows_beta = []
present_targets = [g for g in TARGET_GROUPS if g in groups_all]
def slope_interaction_tests(grp_df, ycol):
    """
    Fit: ΔBAG ~ 1 + V_resid + Gt + Gi + V_resid:Gt + V_resid:Gi  (bilinguals = ref).
    Returns per-group slopes (dΔBAG/dmL) and interaction p-values.
    """
    sub = grp_df.copy()
    sub = sub[sub["group_std"].isin(present_targets)]
    sub = sub[[ycol, VOL_COL, "group_std"]].dropna()
    if len(sub) < max(MIN_N, 6):
        return None
    # predictors and outcome
    x  = pd.to_numeric(sub[VOL_COL], errors="coerce").values.astype(float)   # V_resid
    y  = pd.to_numeric(sub[ycol],    errors="coerce").values.astype(float)   # ΔBAG
    Gt = (sub["group_std"].astype(str) == "translators").astype(float).values
    Gi = (sub["group_std"].astype(str) == "interpreters").astype(float).values
    m = np.isfinite(x) & np.isfinite(y)
    x, y, Gt, Gi = x[m], y[m], Gt[m], Gi[m]
    n = len(x)
    if n < max(MIN_N, 6) or np.nanstd(x) == 0:
        return None
    X = np.column_stack([np.ones(n), x, Gt, Gi, x*Gt, x*Gi])  # [1, x, Gt, Gi, x:Gt, x:Gi]
    try:
        fit = sm.OLS(y, X).fit()
    except Exception:
        return None
    beta = fit.params
    pval = fit.pvalues
    # Group-specific slopes (BAG-years per mL)
    slope_bi = beta[1]
    slope_tr = beta[1] + beta[4]
    slope_in = beta[1] + beta[5]
    return {
        "slope_ref_bilinguals": slope_bi,
        "slope_translators": slope_tr,
        "slope_interpreters": slope_in,
        "p_interaction_trans_vs_bi": pval[4] if len(pval) > 4 else np.nan,
        "p_interaction_interp_vs_bi": pval[5] if len(pval) > 5 else np.nan,
        "n": n
    }
for roi_name, grp_roi in merged.groupby("roi_name"):
    for method in delta_cols:
        res = slope_interaction_tests(grp_roi.assign(group_std=grp_roi["group_std"]), method)  # method is now ycol (ΔBAG)
        if res is None:
            pair_rows_beta.append({
                "roi_name": roi_name, "method": method,
                "slope_ref_bilinguals": np.nan,
                "slope_translators": np.nan,
                "slope_interpreters": np.nan,
                "p_interaction_trans_vs_bi": np.nan,
                "p_interaction_interp_vs_bi": np.nan,
                "n": int(grp_roi[[VOL_COL, method, "group_std"]].dropna().shape[0])
            })
        else:
            pair_rows_beta.append({"roi_name": roi_name, "method": method, **res})
df_beta_diff = pd.DataFrame(pair_rows_beta).sort_values(["roi_name","method"]).reset_index(drop=True)
# FDR on interaction p-values (per method)
print("Applying FDR to interaction tests...")
df_beta_diff["p_trans_vs_bi_fdr"]  = np.nan
df_beta_diff["p_interp_vs_bi_fdr"] = np.nan
for m, idx in df_beta_diff.groupby("method").groups.items():
    idx = list(idx)
    for col_src, col_out in [("p_interaction_trans_vs_bi", "p_trans_vs_bi_fdr"),
                             ("p_interaction_interp_vs_bi", "p_interp_vs_bi_fdr")]:
        pvals = df_beta_diff.loc[idx, col_src].values
        mask = np.isfinite(pvals)
        if mask.sum() > 0:
            _, p_adj, _, _ = multipletests(pvals[mask], alpha=ALPHA, method="fdr_bh")
            df_beta_diff.loc[np.array(idx)[mask], col_out] = p_adj
beta_diff_csv = OUTPUT_DIR_BETA / "roi_volume_vs_delta_between_group_SLOPE_tests_residICV_cv5.csv"
df_beta_diff.to_csv(beta_diff_csv, index=False)
print(f"Saved between-group slope tests (with FDR) -> {beta_diff_csv}")
# 3) Heatmaps of slopes (β1) for target groups
print("Writing NIfTI heatmaps from ROI slopes for bilinguals/translators/interpreters...")
seg_img = nib.load(str(TEMPLATE_SEG))
seg_data = seg_img.get_fdata().astype(int)
roi_map_df = merged.drop_duplicates(subset=["roi_name"])[["roi_name", "label_id"]]
roi_to_label = dict(zip(roi_map_df["roi_name"], roi_map_df["label_id"]))
def write_roi_map(value_map, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vol = np.zeros_like(seg_data, dtype=np.float32)
    for roi_name, val in value_map.items():
        lab = roi_to_label.get(roi_name, None)
        if lab is None or val is None or np.isnan(val):
            continue
        vol[seg_data == int(lab)] = float(val)
    nib.save(nib.Nifti1Image(vol, affine=seg_img.affine, header=seg_img.header), str(out_path))
# Only for target groups; each group gets its own subfolder under the new beta path
for method in delta_cols:
    for g in TARGET_GROUPS:
        if g not in groups_all:
            continue
        sub = df_beta[(df_beta["method"] == method) & (df_beta["group"] == g)]
        beta_map = dict(zip(sub["roi_name"], sub["beta"]))
        q_map    = dict(zip(sub["roi_name"], sub["p_fdr"]))  # FDR-adjusted p within that group
        group_dir = OUTPUT_DIR_BETA / g
        group_dir.mkdir(exist_ok=True, parents=True)
        # Raw beta
        fname_b = f"heatmap_beta_{method}_group-{g}.nii.gz"
        write_roi_map(beta_map, group_dir / fname_b)
        print("Saved", group_dir / fname_b)
        # Beta masked by FDR q < 0.05
        masked = {
            roi: (b if (q is not None and np.isfinite(q) and q < ALPHA) else 0.0)
            for roi, b in beta_map.items()
            for q in [q_map.get(roi, np.nan)]
        }
        fname_mask = f"heatmap_beta_if_q_lt_0.05_{method}_group-{g}.nii.gz"
        write_roi_map(masked, group_dir / fname_mask)
        print("Saved", group_dir / fname_mask)
print("✅ Slope (beta) pipeline complete.")
print(f"Outputs in: {OUTPUT_DIR_BETA} (plus subfolders per group)")
