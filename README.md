# NSCLC Metagenomics Immunotherapy ML

Code for the manuscript *Leveraging Multi-Cancer Metagenomic Gut Microbiome Data to
Predict Immunotherapy Efficacy in Non-Small Cell Lung Cancer* by Pallachulla,
Kouznetsova, Kesari, and Tsigelny (submitted to *Clinical Lung Cancer*).

## What this does
We pooled four public discovery cohorts (Derosa, Lee, McCulloch, Frankel; n = 684) and
selected gut species that differ between responders (R) and non-responders (NR) in the
same direction across several cohorts. A classifier (Random Forest, XGBoost, SVM, or
L1 Logistic Regression) is trained on those species and tested on a held-out NSCLC
cohort (Routy, n = 35). A second script explains the XGBoost model with TreeSHAP.

The pipeline tunes thresholds and hyperparameters with 10-fold cross-validation on the
discovery set, then evaluates once on the external cohort, with an optional label
permutation test.

## Contents
| File | Purpose |
| ---- | ---- |
| `classification_pipeline.py` | Feature selection, model training and tuning, external validation |
| `treeshap_analysis.py` | TreeSHAP importance analysis on the validation cohort |
| `LICENSE` | MIT |

## Inputs
The data are public and not included here (see the paper's Data and Code Availability
section). The scripts expect preprocessed CSVs:

- **Discovery tables**, one per prevalence threshold (0% and 10-50% in 5% steps,
  named `species_unfiltered.csv`, `species_prev10.csv`, ... `species_prev50.csv`).
  Columns: `Run`, `Cohort` (`Derosa`, `Frankel`, `McCulloch`, or `Lee`), `Class`
  (`R` or `NR`), then one column per species. Every file must contain the same patients.
- **Validation table**: the Routy cohort with a `Class` column and the same species
  column names. Markers missing from it are zero-filled and listed in the console.

Upstream preprocessing is not included: Galaxy (KneadData, Bowtie2, MetaPhlAn4) for
cohorts distributed as raw FASTQ files, then undersampling, MMUPHin batch correction,
and prevalence filtering in R 4.6.0.

## Setup
Python 3.10:
```
pip install numpy==2.2.6 pandas==2.3.1 scipy==1.15.3 scikit-learn==1.7.2 xgboost==3.2.0 matplotlib==3.10.9 shap==0.49.0
```

## Usage
**Pipeline.** Edit the paths at the top of `classification_pipeline.py`
(`PREVALENCE_DIR`, `ROUTY_TEST_PATH`, and the output folder, which is set twice inside
`evaluate_external` and must already exist). Set `MODEL_TO_RUN` to `"rf"`,
`"xgboost"`, `"svc"`, or `"logreg"` and run it once per model:
```
python classification_pipeline.py
```
The grid searches use all CPU cores and can take a while. Set
`RUN_PERMUTATION_TEST = True` to enable the permutation test (off by default).

**TreeSHAP.** After an XGBoost run, copy just the marker names from the "Winning
config's marker features" section of the console log into a text file, one per line.
In `treeshap_analysis.py`, set `MODEL_TYPE`, `DISCOVERY_DATA_PATH` (the discovery table
at the winning prevalence level, e.g. `species_prev10.csv`), `VALIDATION_DATA_PATH`,
`BIOMARKER_LIST_PATH`, and `OUTPUT_DIR`. The validation CSV must contain every marker
column, so add the zero-filled ones. Then run:
```
python treeshap_analysis.py
```
`XGB_PARAMS` holds the final hyperparameters from Supplementary Table S2.

## Outputs
| Output | Figure/Table |
|---|---|
| ROC curve PNG per model (`<model>_Routy_roc_curve.png`) | Figure 1 |
| Permutation-test PNG (`<model>_Routy_permutation_test.png`) | Figure 6 |
| Console log (thresholds, hyperparameters, survivor-count grids, top-10 biomarkers) | Supp. Tables S1-S4 |
| `top10_shap_bar.png` | Figure 7 |
| `top10_shap_beeswarm.png` | Supp. Figure S2 |

Figures 2-5, Supplementary Tables S1-S4, and Supplementary Figure S1 were plotted separately using data printed to
the console by these scripts. That plotting code is not included.

## Notes
- Seeds: 1 for models and CV folds, 123 for the permutation test.
- Marker selection uses all discovery samples before cross-validation, so discovery CV
  AUCs can be optimistic. The external cohort is the primary evaluation.

## License
MIT. See `LICENSE`.

## Contact
Igor F. Tsigelny, itsigeln@ucsd.edu
