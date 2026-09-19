# EIGCM manuscript source data

This package organizes the numerical data associated with the figures and tables in
`EIGCM_manuscript_results_redesigned V1_corrected_DGM.docx`.

## Contents

- `EIGCM_manuscript_data_summary.xlsx`: formatted index and summary tables.
- `csv/`: machine-readable data files named by manuscript figure or table number.

## Conventions

- `DGP` is dimensionless.
- `Ev` is vertical eye illuminance in lux.
- The glare-event threshold is DGP >= 0.38.
- Missing Raytraverse High results for C2 are blank.
- Method names are standardized as EIGCM (MF1), Jones dcglare,
  Raytraverse Balanced, and Raytraverse High.

## Important checks

1. The final Fig. 10 plotting script contains values described in the script as
   approximate visual readings. The CSV files in this package preserve the exact
   analysis metrics and should be used for any redraw.
2. The final Fig. 17 redrawing script generates a synthetic scatter cloud while
   displaying the original summary statistics. The raw saturation-term data are
   included here and should be used to regenerate the figure for publication.
3. Table 5 combines baseline values matching the pooled C2 two-view analysis with
   dense-sampling values from a 4,376-record single-view analysis. The values are
   retained as reported, but the compared scopes should be reconciled before the
   final submission.
4. The manuscript text reports a maximum EIGCM 95th-percentile viewpoint RMSE of
   0.0316, while the current aligned C1 dataset gives 0.0325 using a linear
   percentile definition. Confirm the intended C1 source/version and percentile
   convention.
5. The underlying directional dataset for Fig. 3 was not identified during this
   curation pass; only the embedded manuscript figure is currently available.

## Code

https://github.com/a200626190111/EIGCM
