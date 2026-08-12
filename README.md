# Association of bilingual language use with brain age

Analysis and reproducibility repository for the manuscript:

**Association of bilingual language use with brain age: MRI evidence from bilinguals, translators, and interpreters**

## Overview

The study examines MRI-derived brain-age measures in three mutually exclusive language-experience groups:

- 44 non-professional bilinguals
- 35 translators
- 26 interpreters

Final analytic sample: **N = 105**.

Six pretrained brain-age models were evaluated. The main outcome is the age-bias-corrected brain age gap. The repository contains the code used for model screening, cross-validated age-bias correction, age–BAG analyses, figure generation, and exploratory ROI-wise structure–brain-age analyses.

### BAG terminology

For clarity in the revised manuscript:

- **BAG_raw**: uncorrected brain age gap = predicted brain age − chronological age
- **BAG_bias**: predicted linear age-bias component
- **BAG_corr**: age-bias-corrected brain age gap = BAG_raw − BAG_bias

Some historical variable names in the analysis files are retained for compatibility with the original scripts. In particular, columns beginning with `Predicted_age_non_BC_` contain raw BAG values rather than absolute predicted ages.

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

## Code

### `code/age_bias_correction_cv5.py`

Reproduces the five-fold cross-validated linear age-bias correction used for the BAG analysis.

Within each training fold:

```text
BAG_raw = alpha + beta × Age + error
BAG_bias = alpha + beta × Age
BAG_corr = BAG_raw − BAG_bias
```

Historical settings:

- 5 folds
- `shuffle=True`
- `random_state=42`

Exact reproduction requires either the original row order or the saved fold assignment.

Example:

```bash
python code/age_bias_correction_cv5.py \
  --input input/brainpad_results_deidentified.xlsx \
  --fold-col cv_fold \
  --output output/brainpad_results_bias_corrected.xlsx
```

### `code/violin_plot_cohonf.py`

Six-model screening and visualization.

Main operations:

- reshapes BAG values across six pretrained models
- fits the Group × Model mixed-effects model with participant random intercept
- runs model-specific one-way ANOVAs
- applies across-model multiplicity correction
- performs within-model Tukey post-hoc comparisons
- calculates omnibus effect sizes
- produces model-wise violin plots

### `code/scatter_plot_regression.R`

Age–BAG regression analyses and supporting demographic/sensitivity plots.

Includes:

- group-specific Age–BAG slopes
- `Age × Group` interaction models
- pairwise slope comparisons using `emmeans::emtrends`
- raw and corrected BAG plots
- age-distribution plots
- predicted-age plots
- sex-related sensitivity analyses

### `code/scatter_plot_violin_plot_save_separately.R`

Produces the manuscript-style combined violin/scatter figures for uncorrected and age-bias-corrected BAG.

### `code/heatmap.nii.gz_bias_corrected.py`

Exploratory ROI-wise structure–brain-age analysis.

Main operations:

- merges subject-level BAG data with FastSurfer DKT+ASEG ROI volumes
- residualizes ROI volumes for available covariates (ICV, age, sex)
- computes within-group Pearson and Spearman associations
- applies FDR correction across ROIs
- performs between-group correlation-difference tests (Fisher r-to-z)
- produces ROI summary tables and NIfTI heatmaps

## Expected input files

Place approved derived files in `input/`.

### `brainpad_results_deidentified.xlsx`

Subject-level analysis table containing demographics, group membership, ICV, six model-specific raw BAG values, and six age-bias-corrected BAG values.

### `roi_volumes_deidentified.csv`

Long-format ROI table containing:

```text
subject_id
label_id
roi_name
volume_ml
```

For the final sample the expected size is 105 participants × 100 ROIs = 10,500 rows.

### FastSurfer label template

A compatible `aparc.DKTatlas+aseg.deep.withCC.mgz` segmentation is required only for generating NIfTI heatmaps.

## Analysis workflow

1. **Prepare derived input data** in `input/`.
2. **Reproduce age-bias correction** with `age_bias_correction_cv5.py` when raw BAG and fold assignments are available.
3. **Run the six-model screen** with `violin_plot_cohonf.py`.
4. **Run Age–BAG analyses** with `scatter_plot_regression.R`.
5. **Generate manuscript figures** with `scatter_plot_violin_plot_save_separately.R`.
6. **Run exploratory ROI analyses** with `heatmap.nii.gz_bias_corrected.py`.
7. Write generated figures and tables to `output/`.

## Statistical interpretation

The six-model analysis of age-bias-corrected BAG is the primary model-screening analysis. Model-specific analyses should be interpreted in the context of the multiplicity-corrected six-model result. BrainAge-specific age-slope and ROI analyses are follow-up/model-specific characterization analyses rather than independent confirmatory tests.

Uncorrected BAG is retained for transparency and comparison with original model outputs but is secondary to age-bias-corrected BAG because raw BAG is susceptible to age-related estimation bias.

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

Required packages include:

```r
install.packages(c(
  "readxl", "dplyr", "tidyr", "ggplot2", "broom",
  "emmeans", "rstatix", "ggpubr", "patchwork"
))
```

## Paths

The original analysis scripts were developed on local/HPC systems and some legacy scripts still contain historical absolute paths. When reproducing the analyses, replace these with the repository paths:

```text
input/brainpad_results_deidentified.xlsx
input/roi_volumes_deidentified.csv
output/
```

The standalone age-bias correction script already accepts input/output paths through command-line arguments.

## Data availability and privacy

This repository contains analysis code but does **not** currently publish raw MRI or participant-level research data.

Only derived, de-identified/pseudonymised analysis files should be considered for public release, and only after confirmation that such sharing is permitted by the study consent, ethics approval, and institutional governance requirements. Until that approval is confirmed, participant-level input files are excluded by `.gitignore`.

## License

Code in this repository is distributed under the MIT License. Data, if shared separately, remain subject to their own ethical and institutional use restrictions.

## Contact

Mingshi Chen  
Amsterdam UMC  
m.chen@amsterdamumc.nl
