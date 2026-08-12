# Input data

This folder is reserved for **derived, de-identified analysis inputs** used by the scripts in `../code/`.

## Expected files

### `brainpad_results_deidentified.xlsx`
Subject-level analysis table for the final analytic sample (N = 105; 44 bilinguals, 35 translators, 26 interpreters). It should contain:

- anonymous participant ID (`MRI code` in the current scripts)
- group
- age
- sex/gender
- FSIQ
- age of L2 acquisition
- LexTALE score
- ICV (`VoxelVolume_mL` / `ICV_ml`)
- six model-specific raw BAG columns
- six five-fold cross-validated age-bias-corrected BAG columns
- ideally the original `cv_fold` assignment for exact reproduction of the age-bias correction

### `roi_volumes_deidentified.csv`
Long-format FastSurfer DKT+ASEG ROI table containing:

- `subject_id`
- `label_id`
- `roi_name`
- `volume_ml`

For the final sample this table contains 105 participants × 100 ROIs = 10,500 rows.

### `aparc.DKTatlas+aseg.deep.withCC.mgz`
Template segmentation used only when generating NIfTI ROI heatmaps. A compatible FastSurfer DKT+ASEG label image may be used instead.

## Data sharing

Participant-level derived files are intentionally not committed by default. Public release should only occur if permitted by the study consent, ethics approval, and institutional data-governance requirements.

The repository `.gitignore` currently prevents accidental commits of `.xlsx`, `.csv`, `.mgz`, `.nii`, and `.nii.gz` files in this folder. Remove the relevant ignore rule only after data-sharing approval has been confirmed.
