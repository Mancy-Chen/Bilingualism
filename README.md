# Association of bilingual language use with brain age

Analysis and reproducibility repository for the manuscript:

**Association of bilingual language use with brain age: MRI evidence from bilinguals, translators, and interpreters**

## Overview

This study examines MRI-derived brain-age measures in three analytic language-experience groups:

- 44 non-professional bilinguals
- 35 translators
- 26 interpreters

Final analytic sample: **N = 105**.

The analytic groups are separate for statistical analysis, but professional training and activities may overlap, particularly between translators and interpreters.

Six pretrained brain-age models were evaluated. The primary brain-age outcome is the five-fold cross-validated age-bias-corrected brain age gap. The repository contains code for Table 1 participant characteristics, cross-validated age-bias correction, six-model group and Age × Group analyses, model-quality assessment, BrainAge-specific follow-up and sensitivity analyses, figure generation, and exploratory ROI-wise structure–brain-age analyses.

## Brain-age notation and variable names

To keep predicted age, uncorrected BAG, and corrected BAG visually distinct, the cleaned reproducibility dataset uses three variable types:

- **PredAge**: predicted brain age in years
- **BAG_uncorr**: uncorrected brain age gap = predicted brain age − chronological age
- **BAG_corr**: five-fold cross-validated age-bias-corrected brain age gap

`BAG_uncorr` corresponds to **BAG_raw** in the manuscript.

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
│   ├── demographics_table1.py
│   ├── age_bias_correction_cv5.py
│   ├── model_quality_metrics.py
│   ├── violin_plot_cohonf.py
│   ├── age_group_sensitivity.R
│   ├── scatter_plot_regression.R
│   ├── scatter_plot_violin_plot_save_separately.R
│   ├── roi_group_models.py
│   └── heatmap.nii.gz_bias_corrected.py
│
└── output/
    └── README.md
```

Participant-level derived data are not committed to GitHub. Deidentified derived data supporting the findings are available through Zenodo; see **Data availability and privacy** below.

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

### `code/demographics_table1.py`

Reproduces the participant-characteristics summary used for Table 1:

- mean ± SD for age, FSIQ, age of L2 acquisition, LexTALE, and ICV
- one-way ANOVA across the three groups for continuous variables
- female/male counts and percentages
- one overall chi-square test for sex distribution
- university degree/current university enrolment when the corresponding indicator is available

The default output is `output/table1_demographics.csv`. If the public analysis table omits a redundant participant-level university indicator, the design-based 100% row can be included explicitly with `--include-design-based-education`; the script reports that this row comes from the predefined study inclusion criterion rather than inferring it from the data.

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

### `code/model_quality_metrics.py`

Computes the model-quality characteristics reported for all six pretrained models:

- mean absolute error (MAE)
- root mean squared error (RMSE)
- mean prediction error
- residual age dependence quantified as the `Age–BAG_uncorr` slope

The output is written to `output/model_quality_metrics.csv`.

### `code/violin_plot_cohonf.py`

Six-model group comparison and visualisation.

Main operations:

- uses either the six `BAG_corr_<Model>` columns or the six `BAG_uncorr_<Model>` columns
- fits the Group × Model mixed-effects model with participant random intercept
- runs model-specific one-way ANOVAs
- applies Benjamini–Hochberg FDR correction across the six model-specific omnibus tests
- performs within-model Tukey HSD comparisons
- calculates omnibus effect sizes
- saves a model-comparison CSV and violin plot to `output/`

Set `USE_CORRECTED = True` for the primary `BAG_corr` analysis.

### `code/age_group_sensitivity.R`

Reproduces the revised age-related analyses:

- `Age × Group` interaction tests separately for all six `BAG_corr` models
- Benjamini–Hochberg FDR correction across the six interaction tests
- common-age-range sensitivity analysis for BrainAge
- Cook's-distance sensitivity analysis for the BrainAge interaction
- interpreter-specific Cook's-distance slope analysis
- interpreter-specific Huber robust regression

### `code/scatter_plot_regression.R`

BrainAge-specific age–BAG follow-up analyses and figures:

- group-specific Age–BAG slopes
- BrainAge `Age × Group` interaction model
- pairwise comparisons of age slopes with `emmeans::emtrends`
- uncorrected and corrected BAG age plots
- the `Age × Sex` sensitivity analysis for BrainAge `BAG_corr`

### `code/scatter_plot_violin_plot_save_separately.R`

Produces manuscript-style BrainAge violin/scatter figures using:

```text
BAG_uncorr_BrainAge
BAG_corr_BrainAge
```

### `code/roi_group_models.py`

Reproduces the revised formal ROI models:

- residualises ROI volumes for `ICV_ml`, age, and sex
- uses sum-to-zero group contrasts
- estimates the average volume–BAG_corr association across groups
- tests the omnibus `volume_resid × group` interaction for each ROI
- applies FDR correction separately across 100 main-effect and interaction tests
- outputs descriptive within-group OLS slopes

### `code/heatmap.nii.gz_bias_corrected.py`

Exploratory ROI-wise structure–brain-age analysis. The default outcome is `BAG_corr_BrainAge`.

The script:

- merges subject-level BAG data with FastSurfer ROI volumes
- residualises ROI volumes for `ICV_ml`, age, and sex
- computes within-group Pearson and Spearman associations
- applies FDR correction across ROIs
- performs Fisher r-to-z between-group correlation comparisons
- fits direct ROI-volume × group interaction models
- optionally generates FDR-masked NIfTI heatmaps when the FastSurfer label image is available

## Analysis workflow

1. Obtain the approved deidentified derived data from Zenodo and place the required analysis inputs in `input/`.
2. Reproduce Table 1 participant characteristics with `demographics_table1.py`.
3. Reproduce or verify the five-fold age-bias correction with `age_bias_correction_cv5.py`.
4. Characterise prediction error and residual age dependence with `model_quality_metrics.py`.
5. Run the primary six-model `BAG_corr` group comparison with `violin_plot_cohonf.py`.
6. Run the six-model `Age × Group` analysis and BrainAge robustness analyses with `age_group_sensitivity.R`.
7. Run BrainAge-specific age-slope and age/sex figure analyses with `scatter_plot_regression.R`.
8. Generate manuscript-style BrainAge figures with `scatter_plot_violin_plot_save_separately.R`.
9. Run formal effect-coded ROI group models with `roi_group_models.py`.
10. Run within-group ROI correlations, Fisher comparisons, and optional heatmaps with `heatmap.nii.gz_bias_corrected.py`.
11. Generated figures and tables are written to `output/`.

## Statistical interpretation

The six-model analysis of **BAG_corr** is the primary basis for inference about group differences, with FDR correction across the six model-level tests.

The six-model `Age × Group` analysis is likewise corrected across the six pretrained models.

BrainAge-specific age-slope, sensitivity, and ROI analyses are follow-up/model-specific characterisation analyses rather than independent confirmatory tests.

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

`MASS`, distributed with standard R installations, is used for Huber robust regression.

## Data availability and privacy

The deidentified derived data supporting the findings of this study are available through Zenodo:

**DOI: [10.5281/zenodo.22109260](https://doi.org/10.5281/zenodo.22109260)**

Raw MRI data are not distributed through this GitHub repository. The GitHub repository contains the analysis and figure-generation code; participant-level derived input files are intentionally not committed to GitHub.

Only deidentified/pseudonymised data approved for sharing should be used with the reproducibility scripts. Data remain subject to their ethical and institutional use requirements.

## License

Code in this repository is distributed under the MIT License. Data shared separately remain subject to their own ethical and institutional use restrictions.

## Contact

Mingshi Chen  
Amsterdam UMC  
m.chen@amsterdamumc.nl
