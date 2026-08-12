# Association of bilingual language use with brain age

Analysis and reproducibility repository for the manuscript:

**Association of bilingual language use with brain age: MRI evidence from bilinguals, translators, and interpreters**

## Overview

This study examines MRI-derived brain-age measures in three mutually exclusive language-experience groups:

- 44 non-professional bilinguals
- 35 translators
- 26 interpreters

Final analytic sample: **N = 105**.

Six pretrained brain-age models were evaluated. The primary brain-age outcome is the age-bias-corrected brain age gap. The repository contains code for five-fold cross-validated age-bias correction, six-model screening, BrainAge-specific age–BAG analyses, figure generation, and exploratory ROI-wise structure–brain-age analyses.

## BAG notation and variable names

To make the three BAG quantities visually distinct, the revised manuscript and repository use the same terminology:

- **BAG_raw**: uncorrected brain age gap = predicted brain age − chronological age
- **BAG_bias**: predicted linear age-bias component
- **BAG_corr**: age-bias-corrected brain age gap = BAG_raw − BAG_bias

The Excel dataset uses model-specific variable names of the form:

```text
BAG_raw_BrainAge
BAG_bias_BrainAge
BAG_corr_BrainAge

BAG_raw_BrainAgeR
BAG_bias_BrainAgeR
BAG_corr_BrainAgeR

BAG_raw_DeepBrainNet
BAG_bias_DeepBrainNet
BAG_corr_DeepBrainNet

BAG_raw_Pyment
BAG_bias_Pyment
BAG_corr_Pyment

BAG_raw_BRAID_WM
BAG_bias_BRAID_WM
BAG_corr_BRAID_WM

BAG_raw_BRAID_GM
BAG_bias_BRAID_GM
BAG_corr_BRAID_GM
```

The previous historical names (`Predicted_age_non_BC_*`, `Predicted_BAG_non_BC_*`, and `delta_cv5_*`) are no longer used in the cleaned reproducibility dataset or analysis code.

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
- BAG_raw, BAG_bias, and BAG_corr for all six models

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

### FastSurfer label template

A compatible `aparc.DKTatlas+aseg.deep.withCC.mgz` file is needed only for generating NIfTI heatmaps.

## Code

### `code/age_bias_correction_cv5.py`

Reproduces the five-fold cross-validated linear age-bias correction.

Within each training fold:

```text
BAG_raw = alpha + beta × Age + error
```

For held-out participants:

```text
BAG_bias = alpha + beta × Age
BAG_corr = BAG_raw − BAG_bias
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

The script reads every `BAG_raw_<Model>` column and writes the corresponding `BAG_bias_<Model>` and `BAG_corr_<Model>` columns.

### `code/violin_plot_cohonf.py`

Six-model screening and visualization.

Main operations:

- uses either the six `BAG_corr_<Model>` columns or six `BAG_raw_<Model>` columns
- fits the Group × Model mixed-effects model with participant random intercept
- runs model-specific one-way ANOVAs
- applies Benjamini–Hochberg FDR correction across the six model-specific omnibus tests
- performs within-model Tukey HSD comparisons
- calculates omnibus effect sizes
- saves a model-screening CSV and violin plot to `output/`

Set `USE_CORRECTED = True` for the primary BAG_corr analysis.

### `code/scatter_plot_regression.R`

BrainAge-specific age–BAG follow-up analyses.

Uses:

```text
BAG_raw_BrainAge
BAG_corr_BrainAge
```

The script performs:

- group-specific Age–BAG slopes
- `Age × Group` interaction models
- pairwise comparisons of age slopes with `emmeans::emtrends`
- BAG_raw and BAG_corr age plots
- the Age × Sex sensitivity analysis for BAG_corr

Results and figures are written to `output/`.

### `code/scatter_plot_violin_plot_save_separately.R`

Produces manuscript-style BrainAge violin/scatter figures using:

```text
BAG_raw_BrainAge
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
- optionally generates FDR-masked NIfTI heatmaps when a compatible FastSurfer label template is available

## Analysis workflow

1. Place approved derived input files in `input/`.
2. Reproduce or verify the age-bias correction with `age_bias_correction_cv5.py`.
3. Run the primary six-model BAG_corr screen with `violin_plot_cohonf.py`.
4. Run BrainAge-specific age–BAG follow-up analyses with `scatter_plot_regression.R`.
5. Generate manuscript-style BrainAge figures with `scatter_plot_violin_plot_save_separately.R`.
6. Run exploratory ROI analyses with `heatmap.nii.gz_bias_corrected.py`.
7. Generated figures and tables are written to `output/`.

## Statistical interpretation

The six-model analysis of **BAG_corr** is the primary model-screening analysis. Model-specific findings should be interpreted in the context of the multiplicity-corrected six-model result.

BrainAge-specific age-slope and ROI analyses are follow-up/model-specific characterization analyses rather than independent confirmatory tests.

**BAG_raw** is retained for transparency and comparison with original model outputs, but it is secondary because uncorrected BAG is susceptible to age-related estimation bias.

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
