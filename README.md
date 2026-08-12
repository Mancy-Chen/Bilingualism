# Association of bilingual language use with brain age

Analysis and reproducibility repository for the manuscript:

**Association of bilingual language use with brain age: MRI evidence from bilinguals, translators, and interpreters**

## Overview

This study examines MRI-derived brain-age measures in three mutually exclusive language-experience groups:

- 44 non-professional bilinguals
- 35 translators
- 26 interpreters

Final analytic sample: **N = 105**.

Six pretrained brain-age models were evaluated. The primary brain-age outcome is the five-fold cross-validated age-bias-corrected brain age gap. The repository contains code for cross-validated age-bias correction, six-model screening, BrainAge-specific age–BAG analyses, figure generation, and exploratory ROI-wise structure–brain-age analyses.

## Brain-age notation and variable names

To keep predicted age, uncorrected BAG, and corrected BAG visually distinct, the cleaned reproducibility dataset uses three variable types:

- **PredAge**: predicted brain age in years
- **BAG_uncorr**: uncorrected brain age gap = predicted brain age − chronological age
- **BAG_corr**: five-fold cross-validated age-bias-corrected brain age gap

The age-bias term used to obtain `BAG_corr` is an intermediate quantity estimated within each training fold. It is not stored in the main subject-level analysis table because it is not used by downstream statistical analyses. The fold-specific intercept and age-slope parameters can be written separately by `age_bias_correction_cv5.py`.

For each of the six models, the Excel file contains:

```text
PredAge_<Model>
BAG_uncorr_<Model>
BAG_corr_<Model>
```

For example:

```text
PredAge_BrainAge
BAG_uncorr_BrainAge
BAG_corr_BrainAge
```

Model suffixes are:

```text
BrainAge
BrainAgeR
DeepBrainNet
Pyment
BRAID_WM
BRAID_GM
```

Historical names such as `Predicted_age_non_BC_*`, `Predicted_BAG_non_BC_*`, `delta_cv5_*`, `BAG_raw_*`, and `BAG_bias_*` are not used in the cleaned reproducibility input.

## Repository structure

```text
Bilingualism/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── input/
│   └── README.md
│
├── code/
│   ├── age_bias_correction_cv5.py
│   ├── violin_plot_cohonf.py
│   ├── scatter_plot_regression.R
│   ├── scatter_plot_violin_plot_save_separately.R
│   └── heatmap.nii.gz_bias_corrected.py
│
└── output/
    └── README.md
```

Participant-level derived data are not committed by default. See `input/README.md` for the expected input files and data-sharing notes.

## Input data

### `input/brainpad_results_deidentified.xlsx`

The subject-level analysis table contains:

- anonymous participant ID (`MRI code`)
- group
- age
- sex/gender
- FSIQ
- age of L2 acquisition (AoA)
- LexTALE score
- educational indicators
- intracranial volume (`ICV_ml`)
- historical five-fold assignment (`cv_fold`)
- `PredAge`, `BAG_uncorr`, and `BAG_corr` for all six models

The workbook also contains a `Data_Dictionary` sheet describing the variables.

### `input/roi_volumes_deidentified.csv`

Long-format FastSurfer DKT+ASEG ROI table containing:

```text
subject_id
label_id
roi_name
volume_ml
```

For the final analytic sample the expected size is 105 participants × 100 ROIs = **10,500 rows**.

### `input/aparc.DKTatlas+aseg.deep.withCC.mgz`

FastSurfer DKT+ASEG label image used only for mapping ROI statistics back into brain space and generating NIfTI heatmaps. The statistical ROI analyses do not require this file.

## Code

### `code/age_bias_correction_cv5.py`

Reproduces the five-fold cross-validated linear age-bias correction.

Within each training fold:

```text
BAG_uncorr = alpha + beta × Age + error
```

For held-out participants, the expected linear age-bias term is estimated from the training-fold coefficients and subtracted from `BAG_uncorr`:

```text
BAG_corr = BAG_uncorr − (alpha + beta × Age)
```

Historical settings:

- 5 folds
- `shuffle=True`
- `random_state=42`

For exact reproduction, use the saved `cv_fold` assignments:

```bash
python code/age_bias_correction_cv5.py \
  --input input/brainpad_results_deidentified.xlsx \
  --fold-col cv_fold \
  --output output/brainpad_results_bias_corrected.xlsx
```

The script reads every `BAG_uncorr_<Model>` column and writes/overwrites the corresponding `BAG_corr_<Model>` column. Fold-specific `alpha` and `beta_age` values are saved in a separate `cv_parameters` sheet.

### `code/violin_plot_cohonf.py`

Six-model screening and visualization.

Main operations:

- uses either the six `BAG_corr_<Model>` columns or the six `BAG_uncorr_<Model>` columns
- fits the Group × Model mixed-effects model with participant random intercept
- runs model-specific one-way ANOVAs
- applies Benjamini–Hochberg FDR correction across the six model-specific omnibus tests
- performs within-model Tukey HSD comparisons
- calculates omnibus effect sizes
- saves a model-screening CSV and violin plot to `output/`

Set `USE_CORRECTED = True` for the primary `BAG_corr` analysis.

### `code/scatter_plot_regression.R`

BrainAge-specific age–BAG follow-up analyses using:

```text
BAG_uncorr_BrainAge
BAG_corr_BrainAge
```

The script performs:

- group-specific Age–BAG slopes
- `Age × Group` interaction models
- pairwise comparisons of age slopes with `emmeans::emtrends`
- uncorrected and corrected BAG age plots
- the Age × Sex sensitivity analysis for `BAG_corr`

Results and figures are written to `output/`.

### `code/scatter_plot_violin_plot_save_separately.R`

Produces manuscript-style BrainAge violin/scatter figures using:

```text
BAG_uncorr_BrainAge
BAG_corr_BrainAge
```

### `code/heatmap.nii.gz_bias_corrected.py`

Exploratory ROI-wise structure–brain-age analysis. The default outcome is:

```text
BAG_corr_BrainAge
```

The script:

- merges subject-level BAG data with FastSurfer ROI volumes
- residualizes ROI volumes for `ICV_ml`, age, and sex
- computes within-group Pearson and Spearman associations
- applies FDR correction across ROIs
- performs Fisher r-to-z between-group correlation comparisons
- fits direct ROI-volume × group interaction models
- optionally generates FDR-masked NIfTI heatmaps when the FastSurfer label image is available

## Analysis workflow

1. Place approved derived input files in `input/`.
2. Reproduce or verify the five-fold age-bias correction with `age_bias_correction_cv5.py`.
3. Run the primary six-model `BAG_corr` screen with `violin_plot_cohonf.py`.
4. Run BrainAge-specific age–BAG follow-up analyses with `scatter_plot_regression.R`.
5. Generate manuscript-style BrainAge figures with `scatter_plot_violin_plot_save_separately.R`.
6. Run exploratory ROI analyses with `heatmap.nii.gz_bias_corrected.py`.
7. Generated figures and tables are written to `output/`.

## Statistical interpretation

The six-model analysis of **BAG_corr** is the primary model-screening analysis. Model-specific findings should be interpreted in the context of the multiplicity-corrected six-model result.

BrainAge-specific age-slope and ROI analyses are follow-up/model-specific characterization analyses rather than independent confirmatory tests.

**BAG_uncorr** is retained for transparency and comparison with the original model output, but it is secondary because uncorrected BAG is susceptible to age-related estimation bias.

## Software

### Python

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

Core packages:

- numpy
- pandas
- scipy
- statsmodels
- scikit-learn
- matplotlib
- seaborn
- nibabel
- openpyxl

### R

R 4.2 or later is recommended.

```r
install.packages(c(
  "readxl", "dplyr", "tidyr", "ggplot2", "broom",
  "emmeans", "rstatix", "ggpubr", "patchwork"
))
```

## Data availability and privacy

This repository contains analysis code but does **not** currently publish raw MRI or participant-level research data.

Only derived, pseudonymised analysis files should be considered for public release, and only after confirmation that sharing is permitted by the study consent, ethics approval, and institutional governance requirements. Until that approval is confirmed, participant-level input files are excluded by `.gitignore`.

## License

Code in this repository is distributed under the MIT License. Data, if shared separately, remain subject to their own ethical and institutional use restrictions.

## Contact

Mingshi Chen  
Amsterdam UMC  
m.chen@amsterdamumc.nl
