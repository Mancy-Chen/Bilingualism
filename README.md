README
===========

Title
-----
Language processing demands affect brain age: analysis code
(MRI evidence from bilinguals, translators, and interpreters)

Overview
--------
This repository contains analysis scripts used for the manuscript:
"Language processing demands affect brain age: MRI evidence from bilinguals, translators, and interpreters"

The code supports three main outputs:
1) Model-wise violin plots for six brain-age models across groups
2) Age–BAG and Age–ΔBAG regression plots and slope comparisons
3) ROI-wise structure–ΔBAG association results and NIfTI heatmaps mapped to atlas label space

Important: This repository does NOT include raw MRI or restricted participant data.
Only derived, de-identified tables should be used locally.

Included scripts
----------------
1) scatter_plot_regression.R
   - Reads a derived Excel table
   - Plots Age vs raw BAG and Age vs bias-corrected ΔBAG by group
   - Fits within-group linear models and Age*Group interaction models
   - Performs slope comparisons (emmeans/emtrends)
   - Saves multiple figures (PNG)

2) violin_plot_cohonD.py
   - Reads a derived Excel table containing outputs from six pretrained brain-age models
   - Produces violin plots by group for each model (raw or bias-corrected)
   - Runs ANOVA and Tukey post-hoc tests
   - Computes effect sizes (Cohen’s d)
   - Saves a final violin plot (PNG)

3) heatmap.nii.gz_bias_corrected.py
   - Reads a long-format ROI volume table (FastSurfer DKT+ASEG) and a brain-age results table
   - Residualizes ROI volumes for covariates (e.g., ICV, Age, Gender, Site if available)
   - Computes within-group Pearson/Spearman correlations between ROI volume residuals and ΔBAG
   - Applies FDR correction across ROIs
   - Runs between-group correlation difference tests (Fisher r-to-z) with FDR correction
   - Saves CSV summaries and NIfTI heatmaps (.nii.gz) aligned to a template label image

Software requirements
---------------------
R (for scatter_plot_regression.R)
- R >= 4.2 recommended
- Packages: readxl, dplyr, tidyr, ggplot2, ggpubr, patchwork, broom, emmeans, rstatix

Install in R:
install.packages(c("readxl","dplyr","tidyr","ggplot2","ggpubr","patchwork","broom","emmeans","rstatix"))

Python (for violin_plot_cohonD.py and heatmap.nii.gz_bias_corrected.py)
- Python 3.10 recommended
- Packages: numpy, pandas, scipy, statsmodels, matplotlib, seaborn, nibabel

Install with pip:
pip install numpy pandas scipy statsmodels matplotlib seaborn nibabel

Recommended folder layout
-------------------------
Create a simple structure like:

repo_root/
  scripts/
    scatter_plot_regression.R
    violin_plot_cohonD.py
    heatmap.nii.gz_bias_corrected.py
  data/
    brainpad_results.xlsx
    brainpad_results_radiomics.xlsx
    DKTatlas_aseg_deep_withCC_long.csv
    aparc.DKTatlas+aseg.deep.withCC.mgz
  outputs/

Notes:
- The scripts currently contain hard-coded file paths (e.g., C:/... or /data/projects/...).
- Before running, edit each script to point to your own local file locations.
- Prefer relative paths such as: data/brainpad_results.xlsx and outputs/

Expected input data (derived)
-----------------------------
A) Excel table with demographics + group + brain-age outputs
   Typical required columns:
   - Age
   - group  (values: bilinguals, translators, interpreters)
   - subject identifier (e.g., "MRI code" or "subject_id")
   - model outputs:
     raw BAG columns (example): Predicted_BAG_non_BC_Brainage
     bias-corrected columns (example): delta_cv5_Predicted_age_non_BC_Brainage
   The exact column names can vary; adjust in scripts if needed.

B) ROI table for heatmap script (CSV)
   - Long-format ROI data from FastSurfer DKT+ASEG
   - Should include subject_id and ROI volume values
   - ROI definition corresponds to FastSurfer aparc.DKTatlas+aseg (DKT cortex + ASEG subcortex; ~100 ROIs)

C) Template label image for mapping heatmaps
   - FastSurfer segmentation label file, e.g. aparc.DKTatlas+aseg.deep.withCC.mgz

How to run
----------
1) Update paths inside the scripts to match your local system.
   - Set input Excel/CSV paths
   - Set output directory paths

2) Run R script:
   In R (from repo root):
   source("scripts/scatter_plot_regression.R")

3) Run Python violin plot:
   python scripts/violin_plot_cohonD.py

4) Run Python ROI heatmap:
   python scripts/heatmap.nii.gz_bias_corrected.py

Outputs
-------
- PNG figures (scatter plots and violin plots), saved by the scripts
- CSV tables summarizing ROI correlations, FDR results, and between-group tests
- NIfTI heatmaps (.nii.gz) for correlation and regression effects mapped to atlas label space

Privacy and sharing
-------------------
- Do not upload raw MRI, NDA-controlled data, or identifiable participant data to GitHub.
- Share only code + instructions + (optional) synthetic/example data.

Contact
-------
Mingshi Chen
m.chen@amsterdamumc.nl
