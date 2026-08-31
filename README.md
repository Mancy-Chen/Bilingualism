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

The reproducibility dataset uses three variable types:

- **PredAge**: predicted brain age in years
- **BAG_uncorr**: uncorrected brain age gap = predicted brain age − chronological age
- **BAG_corr**: five-fold cross-validated age-bias-corrected brain age gap

`BAG_uncorr` corresponds to **BAG_raw** in the manuscript.

The deposited dataset contains the final `BAG_corr` values used in the reported analyses. The age-bias term itself is an intermediate quantity estimated within each training fold and is not required by the downstream analyses.

For each of the six models, the Excel file contains:

```text
PredAge_<Model>
BAG_uncorr_<Model>
BAG_corr_<Model>
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

Participant-level derived data are not committed to GitHub. Deidentified derived data supporting the findings are available through Zenodo.

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

Reproduces the participant-characteristics summary used for Table 1, including one-way ANOVAs for continuous characteristics and the overall chi-square test for sex distribution.

### `code/age_bias_correction_cv5.py`

Implements five-fold cross-validated linear age-bias correction using the documented settings:

- 5 folds
- `shuffle=True`
- `random_state=42`

Within each training fold:

```text
BAG_uncorr = alpha + beta × Age + error
```

For held-out participants:

```text
BAG_corr = BAG_uncorr − (alpha + beta × Age)
```

The deposited dataset already contains the final `BAG_corr` values used for all manuscript analyses. The script is provided to document and regenerate the correction procedure from `BAG_uncorr` using the stated settings.

```bash
python code/age_bias_correction_cv5.py \
  --input input/brainpad_results_deidentified.xlsx \
  --output output/brainpad_results_bias_corrected.xlsx
```

### `code/model_quality_metrics.py`

Computes MAE, RMSE, mean prediction error, and the residual `Age–BAG_uncorr` slope for all six pretrained models.

### `code/violin_plot_cohonf.py`

Runs the six-model group comparison, model-specific one-way ANOVAs, Benjamini–Hochberg FDR correction, Tukey HSD comparisons, effect sizes, and violin plots.

### `code/age_group_sensitivity.R`

Reproduces the six-model `Age × Group` interaction tests with FDR correction and the BrainAge common-age-range, Cook's-distance, and Huber robust sensitivity analyses.

### `code/scatter_plot_regression.R`

Reproduces BrainAge-specific age–BAG follow-up analyses, slope comparisons, figures, and the `Age × Sex` sensitivity analysis.

### `code/scatter_plot_violin_plot_save_separately.R`

Produces manuscript-style BrainAge violin and scatter figures.

### `code/roi_group_models.py`

Reproduces the revised formal ROI models, including average volume–BAG_corr associations, omnibus volume × group interactions, FDR correction, and descriptive within-group OLS slopes.

### `code/heatmap.nii.gz_bias_corrected.py`

Runs within-group ROI correlations, Fisher r-to-z comparisons, direct ROI-volume × group interaction models, and optional FDR-masked NIfTI heatmaps.

## Analysis workflow

1. Obtain the deidentified derived data from Zenodo and place the required files in `input/`.
2. Reproduce Table 1 with `demographics_table1.py`.
3. Use the deposited `BAG_corr` values for the manuscript analyses; `age_bias_correction_cv5.py` documents the correction procedure.
4. Run `model_quality_metrics.py`.
5. Run the six-model group analysis with `violin_plot_cohonf.py`.
6. Run `age_group_sensitivity.R`.
7. Run `scatter_plot_regression.R` and `scatter_plot_violin_plot_save_separately.R`.
8. Run `roi_group_models.py` and `heatmap.nii.gz_bias_corrected.py`.

## Statistical interpretation

The six-model analysis of **BAG_corr** is the primary basis for inference about group differences, with FDR correction across the six model-level tests. The six-model `Age × Group` analysis is likewise corrected across the six pretrained models. BrainAge-specific age-slope, sensitivity, and ROI analyses are follow-up/model-specific characterisation analyses rather than independent confirmatory tests.

## Software

Python 3.10 or later is recommended. Install Python dependencies with:

```bash
pip install -r requirements.txt
```

R 4.2 or later is recommended. The R scripts use `readxl`, `dplyr`, `tidyr`, `ggplot2`, `broom`, `emmeans`, `rstatix`, `ggpubr`, `patchwork`, and `MASS`.

## Data availability and privacy

The deidentified derived data supporting the findings of this study are available through Zenodo:

**DOI: [10.5281/zenodo.22109260](https://doi.org/10.5281/zenodo.22109260)**

Raw MRI data are not distributed through this GitHub repository. The GitHub repository contains the analysis and figure-generation code; participant-level derived input files are intentionally not committed to GitHub.

## License

Code in this repository is distributed under the MIT License. Data shared separately remain subject to their own ethical and institutional use restrictions.

## Contact

Mingshi Chen  
Amsterdam UMC  
m.chen@amsterdamumc.nl
