# Input data

This folder is reserved for **derived, de-identified/pseudonymised analysis inputs** used by the scripts in `../code/`.

## Expected files

### `brainpad_results_deidentified.xlsx`

Subject-level analysis table for the final analytic sample:

- N = 105
- 44 bilinguals
- 35 translators
- 26 interpreters

Core participant variables include:

- `MRI code` — anonymous participant ID (P001–P105)
- `Gender`
- `group`
- `Age`
- `FSIQ`
- `AoA`
- `LexTale`
- `ICV_ml`
- `cv_fold`

Brain-age variables use the final reproducibility notation:

```text
PredAge_<Model>
BAG_uncorr_<Model>
BAG_corr_<Model>
```

Available model suffixes are:

```text
BrainAge
BrainAgeR
DeepBrainNet
Pyment
BRAID_WM
BRAID_GM
```

For example:

```text
PredAge_BrainAge
BAG_uncorr_BrainAge
BAG_corr_BrainAge
```

Definitions:

- `PredAge` = predicted brain age in years
- `BAG_uncorr` = predicted brain age − chronological age before age-bias correction
- `BAG_corr` = five-fold cross-validated age-bias-corrected BAG

The age-bias component itself is an intermediate quantity and is not stored in the main analysis table because downstream analyses do not use it.

The workbook also contains a `Data_Dictionary` sheet.

### `roi_volumes_deidentified.csv`

Long-format FastSurfer DKT+ASEG ROI table containing:

```text
subject_id
label_id
roi_name
volume_ml
```

For the final sample this table contains 105 participants × 100 ROIs = **10,500 rows**.

### `aparc.DKTatlas+aseg.deep.withCC.mgz`

FastSurfer DKT+ASEG label image used only when generating NIfTI ROI heatmaps. It is not needed for the statistical ROI calculations themselves.

## Data sharing

Participant-level derived files are intentionally not committed by default. Public release should only occur if permitted by the study consent, ethics approval, and institutional data-governance requirements.

The repository `.gitignore` prevents accidental commits of `.xlsx`, `.csv`, `.mgz`, `.nii`, and `.nii.gz` files in this folder unless those rules are intentionally changed after data-sharing approval.
