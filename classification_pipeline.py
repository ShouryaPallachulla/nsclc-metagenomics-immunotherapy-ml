"""
Cross-cohort microbiome biomarker discovery, machine learning model training, and external validation.
Code accompanying the manuscript submitted to Clinical Lung Cancer.

The pipeline looks for species that differ between responders (R) and
non-responders (NR) in the same direction across several discovery cohorts,
trains a machine learning classifier on those species, and tests it on a held-out cohort (Routy).

  Stage A: select the optimal prevalence filter and marker-selection thresholds
           (p-value threshold, minimum number of consistent cohorts) by 10-fold CV AUC.
  Stage B: tune the model's hyperparameters on the Stage A marker set.
  Stage C: evaluate the final model on the external cohort.

Usage: set MODEL_TO_RUN below and run the script once per model
("rf", "xgboost", "svc", "logreg").

Inputs:  one multi-cohort discovery CSV per prevalence threshold (columns: Run, Cohort,
         Class, then one column per species with relative abundances) and one
         external validation CSV.
Outputs: console log, ROC curve PNG, and an optional permutation-test PNG

Dependencies: numpy v2.2.6, pandas v2.3.1, scipy v1.15.3, scikit-learn v1.7.2, xgboost v3.2.0, matplotlib v3.10.9
"""


import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")  # no display needed; figures are saved to file
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve)

from xgboost import XGBClassifier


class CLRPseudocountTransformer(BaseEstimator, TransformerMixin):
    """Centered log-ratio (CLR) transform with a pseudocount for zeros.

    Negatives are clipped to 0, zeros are replaced by half the smallest positive
    value seen in the training data, and each sample is log-transformed and
    centered on its own mean. The pseudocount is learned in fit() only, so
    held-out samples never influence it. Used for logistic regression only.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_negative = int(np.sum(X < 0))  # kept for diagnostics
        X_clipped = np.clip(X, 0, None)
        positive_vals = X_clipped[X_clipped > 0]
        if positive_vals.size == 0:
            raise ValueError(
                "CLRPseudocountTransformer.fit: no strictly positive values found in this "
                "training fold after clipping negatives to 0 - cannot derive a pseudocount."
            )
        self.pseudocount_ = float(positive_vals.min() / 2.0)
        self.n_negative_clipped_in_fit_ = n_negative
        return self

    def transform(self, X):
        if not hasattr(self, "pseudocount_"):
            raise RuntimeError("CLRPseudocountTransformer.transform called before fit.")
        X = np.asarray(X, dtype=float)
        X_clipped = np.clip(X, 0, None)
        X_replaced = np.where(X_clipped <= 0, self.pseudocount_, X_clipped)
        log_X = np.log(X_replaced)
        clr = log_X - log_X.mean(axis=1, keepdims=True)  # center each sample on its mean log value
        return clr

# CONFIG: WHICH MODEL THIS RUN EVALUATES
# One of: "rf", "xgboost", "svc", "logreg". Run this script once per model
# (four separate runs total) - results do not depend on run order, and
# each run's Stage A / Stage B search is entirely independent of the others.
MODEL_TO_RUN = "rf"

# Cohorts used for discovery and training. Routy is held out for external validation.
DISCOVERY_COHORTS = ["Derosa", "Frankel", "McCulloch", "Lee"]



# A species passes the filter only if it is detected in at least this fraction of
# samples (0.0 = unfiltered). Filtering was done upstream using R; there is one file per threshold.
PREVALENCE_THRESHOLDS = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
PREVALENCE_DIR = r"path/to/prevalence_files"  # placeholder - set to your local directory
PREVALENCE_FILE_PATHS = {
    0.0: os.path.join(PREVALENCE_DIR, "species_unfiltered.csv"),
    0.10: os.path.join(PREVALENCE_DIR, "species_prev10.csv"),
    0.15: os.path.join(PREVALENCE_DIR, "species_prev15.csv"),
    0.20: os.path.join(PREVALENCE_DIR, "species_prev20.csv"),
    0.25: os.path.join(PREVALENCE_DIR, "species_prev25.csv"),
    0.30: os.path.join(PREVALENCE_DIR, "species_prev30.csv"),
    0.35: os.path.join(PREVALENCE_DIR, "species_prev35.csv"),
    0.40: os.path.join(PREVALENCE_DIR, "species_prev40.csv"),
    0.45: os.path.join(PREVALENCE_DIR, "species_prev45.csv"),
    0.50: os.path.join(PREVALENCE_DIR, "species_prev50.csv"),
}

# Stage A search ranges: per-cohort p-value thresholds, and the minimum number of
# cohorts in which a species must be significant in the same direction.
P_THRESHOLD_CANDIDATES = [0.05, 0.10, 0.15, 0.20]
MIN_CONSISTENT_COHORTS_CANDIDATES = [2, 3, 4]
# Combinations that select fewer markers than this are skipped.
MIN_SURVIVORS_FOR_VIABLE_COMBO = 10

# Shared CV configuration
NUM_FOLDS = 10
SEED = 1
KFOLD = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)  # ONE object, reused everywhere
# (so Stage A and Stage B use the same folds)

# Label permutation test (external set only)
RUN_PERMUTATION_TEST = False
N_PERMUTATIONS = 1000
PERMUTATION_SEED = 123

# Per-model hyperparameter grids (Stage B)
# "clf__" prefixes address the classifier step inside a Pipeline.
# RF: 9 x 9 x 3 x 2 x 3 = 1458 combinations
RF_PARAM_GRID = {
    "n_estimators": list(range(100, 501, 50)),
    "max_depth": list(range(3, 20, 2)),
    "max_features": ["log2", "sqrt", 0.2],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [1, 3, 5],
}

# XGBoost: 9 x 3 x 4 x 2 x 2 = 432 combinations
XGB_PARAM_GRID = {
    "n_estimators": list(range(100, 501, 50)),
    "max_depth": list(range(3, 10, 3)),
    "learning_rate": [0.025, 0.05, 0.075, 0.1],
    "subsample": [0.5, 1.0],
    "colsample_bytree": [0.5, 1.0], 
}

# SVC: 4 x 6 = 24 combinations
SVC_PARAM_GRID = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": [0.0001, 0.001, 0.01, 0.1, 1.0, 10],
}

# L1-regularized Logistic Regression: 7 x 2 = 14 combinations
LOGREG_PARAM_GRID = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "clf__solver": ["liblinear", "saga"],
}

# Model configs. Preprocessing differs by model: none for the tree-based models,
# standardization for SVC, and CLR plus standardization for logistic regression.
# Estimators use n_jobs=1 because parallelism is handled at the CV level.
MODEL_CONFIGS = {
    "rf": {
        "display_name": "RandomForest",
        "estimator_fn": lambda: RandomForestClassifier(random_state=SEED, n_jobs=1),
        "pipeline_steps": [],
        "param_grid": RF_PARAM_GRID,
    },
    "xgboost": {
        "display_name": "XGBoost",
        "estimator_fn": lambda: XGBClassifier(tree_method="hist", eval_metric="logloss",
                                               random_state=SEED, n_jobs=1),
        "pipeline_steps": [],
        "param_grid": XGB_PARAM_GRID,
    },
    "svc": {
        "display_name": "SVC (RBF)",
        "estimator_fn": lambda: SVC(kernel="rbf", random_state=SEED),
        "pipeline_steps": [("scaler", StandardScaler())],
        "param_grid": SVC_PARAM_GRID,
    },
    "logreg": {
        "display_name": "Logistic Regression",
        "estimator_fn": lambda: LogisticRegression(penalty="l1", solver="liblinear",
                                                    random_state=SEED, max_iter=10000),
        "pipeline_steps": [("clr", CLRPseudocountTransformer()), ("scaler", StandardScaler())],
        "param_grid": LOGREG_PARAM_GRID,
    },
}

if MODEL_TO_RUN not in MODEL_CONFIGS:
    raise ValueError(f"MODEL_TO_RUN must be one of {list(MODEL_CONFIGS.keys())}, got '{MODEL_TO_RUN}'")


ROUTY_TEST_PATH = r"path/to/routy_external_validation.csv"  # placeholder - set to your local path


RUN_COL = "Run"
COHORT_COL = "Cohort"
CLASS_COL = "Class"  # "R" (responder) or "NR" (non-responder)
METADATA_COLS = [RUN_COL, COHORT_COL, CLASS_COL]


# Print a warning if more than this fraction of the marker panel is missing from
# the external data and has to be zero-filled.
MAX_ZEROFILL_FRACTION_WARN = 0.25




## Step 0 helper: load + validate one prevalence level's discovery file
def load_and_validate_discovery_file(path, label):
    """Load one discovery table and check it is usable.

    Checks that the required columns exist, column names and Run IDs are unique,
    every discovery cohort is present, and Class is only "R" or "NR" with no
    missing values. Returns the table and the list of feature columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label}: file not found at {path}."
        )
    df_level = pd.read_csv(path)

    for col in [RUN_COL, COHORT_COL, CLASS_COL]:
        if col not in df_level.columns:
            raise ValueError(f"{label}: expected column '{col}' not found. "
                              f"Columns present: {list(df_level.columns)}")

    if df_level.columns.duplicated().any():
        dupe_names = df_level.columns[df_level.columns.duplicated()].tolist()
        raise ValueError(f"{label}: duplicate column names found: {dupe_names}")

    if df_level[RUN_COL].duplicated().any():
        dupe_runs = df_level.loc[df_level[RUN_COL].duplicated(), RUN_COL].tolist()
        raise ValueError(f"{label}: duplicate Run IDs found: {dupe_runs}")

    present_cohorts = set(df_level[COHORT_COL].unique())

    missing_cohorts = set(DISCOVERY_COHORTS) - present_cohorts
    if missing_cohorts:
        raise ValueError(f"{label}: expected cohort(s) not found at all: {sorted(missing_cohorts)}")

    unexpected_class = set(df_level[CLASS_COL].dropna().unique()) - {"R", "NR"}
    if unexpected_class:
        raise ValueError(f"{label}: unexpected Class value(s): {unexpected_class}. Expected only 'R'/'NR'.")
    if df_level[CLASS_COL].isna().any():
        raise ValueError(f"{label}: {df_level[CLASS_COL].isna().sum()} row(s) have a missing Class value.")

    feature_cols_level = [c for c in df_level.columns if c not in METADATA_COLS]
    return df_level, feature_cols_level


def compute_direction_pvalues(df_level, feature_cols_level):
    """Run a two-sided Mann-Whitney U test (R vs NR) for every feature in every cohort.

    Direction is "up" if the median is higher in R than NR, otherwise "down"
    (ties count as "down"). P-values are not adjusted for multiple testing.
    Features that can't be tested in a cohort are stored as None.
    Returns {feature: {cohort: (direction, p_value) or None}}.
    """
    results = {feat: {} for feat in feature_cols_level}
    for cohort in DISCOVERY_COHORTS:
        sub = df_level[df_level[COHORT_COL] == cohort]
        r_sub = sub[sub[CLASS_COL] == "R"]
        nr_sub = sub[sub[CLASS_COL] == "NR"]
        print(f"    Cohort {cohort}: {len(sub)} patients (R={len(r_sub)}, NR={len(nr_sub)})")


        for feat in feature_cols_level:
            r_vals = r_sub[feat].dropna().to_numpy()
            nr_vals = nr_sub[feat].dropna().to_numpy()
            if len(r_vals) == 0 or len(nr_vals) == 0:
                results[feat][cohort] = None
                continue
            try:
                _, p_value = mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
            except ValueError:
                # e.g. all values identical, so the test can't be computed
                results[feat][cohort] = None
                continue
            direction = "up" if np.median(r_vals) > np.median(nr_vals) else "down"
            results[feat][cohort] = (direction, p_value)
    return results


def markers_from_results(results, p_threshold, min_cohorts):
    """Keep features that are significant in the same direction in at least min_cohorts cohorts.

    Features that pass in both directions at once are dropped as contradictory.
    Returns the selected feature names and the per-feature counts of significant
    "up" and "down" cohorts.
    """
    markers = []
    direction_counts = {}
    for feat, cohort_data in results.items():
        tested = [v for v in cohort_data.values() if v is not None]
        up_sig = sum(1 for d, p in tested if d == "up" and p < p_threshold)
        down_sig = sum(1 for d, p in tested if d == "down" and p < p_threshold)
        direction_counts[feat] = {"up": up_sig, "down": down_sig}
        up_pass = up_sig >= min_cohorts
        down_pass = down_sig >= min_cohorts
        if up_pass and down_pass:
            continue
        if up_pass or down_pass:
            markers.append(feat)
    return markers, direction_counts


def top_biomarkers_overall(results, p_threshold, min_cohorts=3, top_n=10):
    """Rank directionally consistent biomarkers by average p-value (reporting only).

    A feature qualifies if it is significant in the same direction in at least
    min_cohorts cohorts. The average p-value is taken over every cohort where the
    feature could be tested, not just the significant ones. This does not affect
    model training.

    Returns a single list of dicts, ordered best (lowest avg_p) first:
        {"feature", "avg_p", "direction", "rank", "consistency_count", "cohorts"}
    """
    candidates = []
    for feat, cohort_data in results.items():
        tested = {cohort: v for cohort, v in cohort_data.items() if v is not None}
        if not tested:
            continue
        avg_p = float(np.mean([p for (_, p) in tested.values()]))
        up_cohorts = sorted(c for c, (d, p) in tested.items() if d == "up" and p < p_threshold)
        down_cohorts = sorted(c for c, (d, p) in tested.items() if d == "down" and p < p_threshold)
        if len(up_cohorts) >= min_cohorts:
            candidates.append({"feature": feat, "avg_p": avg_p, "direction": "up",
                                "consistency_count": len(up_cohorts), "cohorts": up_cohorts})
        if len(down_cohorts) >= min_cohorts:
            candidates.append({"feature": feat, "avg_p": avg_p, "direction": "down",
                                "consistency_count": len(down_cohorts), "cohorts": down_cohorts})

    candidates.sort(key=lambda r: r["avg_p"])
    top_records = candidates[:top_n]
    for rank, rec in enumerate(top_records, start=1):
        rec["rank"] = rank
    return top_records


def print_top_biomarkers(top_records, p_threshold, min_cohorts):
    print(f"Top {len(top_records)} biomarkers overall, "
          f"consistent direction in >= {min_cohorts} cohorts, significance defined as "
          f"p < {p_threshold})")
    if not top_records:
        print(f"  No biomarkers found with consistent direction in >= {min_cohorts} cohorts.\n")
        return
    for rec in top_records:
        print(f"  #{rec['rank']}  {rec['feature']}: avg p-value={rec['avg_p']:.4g}, "
              f"direction={rec['direction']}, consistency count={rec['consistency_count']} cohort(s)")
        print(f"        Consistent signal in: {', '.join(rec['cohorts'])}")
    print()


## Model-building helper: raw estimator for RF/XGBoost, or a
## Pipeline(preprocessing steps..., clf) for SVC/LogReg.
def build_estimator(model_key):
    cfg = MODEL_CONFIGS[model_key]
    steps = list(cfg["pipeline_steps"])
    if steps:
        return Pipeline(steps + [("clf", cfg["estimator_fn"]())])
    return cfg["estimator_fn"]()


def get_scores(fitted_estimator, X):
    """Return a continuous score for the positive class (R), using predicted
    probabilities when the model has them and the decision function otherwise."""
    if hasattr(fitted_estimator, "predict_proba"):
        return fitted_estimator.predict_proba(X)[:, 1]
    return fitted_estimator.decision_function(X)



def compute_metrics(y_true, y_score):
    return {
        "auc": roc_auc_score(y_true, y_score),
    }


def run_permutation_test(y_true, y_score, n_permutations, seed):
    """Test whether the observed AUC beats chance by shuffling the labels.

    Scores stay fixed while labels are permuted to build a null AUC distribution.
    The one-sided p-value is (null AUCs >= observed + 1) / (n_permutations + 1),
    so it can never be exactly zero.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    observed_auc = roc_auc_score(y_true, y_score)
    null_aucs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled_labels = rng.permutation(y_true)
        null_aucs[i] = roc_auc_score(shuffled_labels, y_score)
    p_value = (np.sum(null_aucs >= observed_auc) + 1) / (n_permutations + 1)
    return observed_auc, null_aucs, p_value


def plot_permutation_distribution(null_aucs, observed_auc, p_value, n_permutations, save_path):
    # Uses the global model_cfg (set in Stage A) for the title.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_aucs, bins=30, color="steelblue", edgecolor="black", alpha=0.8,
            label=f"Null distribution (n={n_permutations} permutations)")
    ax.axvline(observed_auc, color="red", linestyle="--", linewidth=2,
               label=f"Observed AUC = {observed_auc:.4f}")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Frequency")
    floor_p = 1 / (n_permutations + 1)  # smallest p-value the test can produce
    p_label = f"p < {floor_p:.4f}" if p_value <= floor_p else f"p = {p_value:.4f}"
    ax.set_title(f"Label permutation test: {model_cfg['display_name']} ({p_label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved permutation-test distribution plot to: {save_path}")


def plot_roc_curve(y_true, y_score, auc_value, label, save_path):
    # Uses the global model_cfg (set in Stage A) for the title.
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"ROC curve (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC curve: {model_cfg['display_name']} on {label} (external validation)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved ROC curve plot to: {save_path}")


#Stage A: prevalence level x threshold-combo search
# For each prevalence file, select markers under every viable (p-threshold,
# min-cohorts) combination and score each marker set by 10-fold CV AUC with
# default hyperparameters. The best-scoring configuration moves on to Stage B.
# Note: marker selection uses all discovery samples before CV, and Stages A and B share
# the same folds, so discovery CV AUC may be biased. Performance is assessed on the
# external cohort.
model_cfg = MODEL_CONFIGS[MODEL_TO_RUN]
print(f"Running model: {model_cfg['display_name']} ({MODEL_TO_RUN})")
print(f"Stage A default estimator: {build_estimator(MODEL_TO_RUN)}\n")

print("Stage A: prevalence level x threshold-combo search\n")

stage_a_records = []
reference_run_set = None
reference_run_set_threshold = None

for threshold in PREVALENCE_THRESHOLDS:
    pct = f"{threshold:.0%}"
    path = PREVALENCE_FILE_PATHS[threshold]
    print(f"Prevalence >= {pct}: {path}")

    df_level, feature_cols_level = load_and_validate_discovery_file(path, f"prevalence >= {pct}")
    print(f"  {len(df_level)} patients, {len(feature_cols_level)} feature column(s) in this file.")

    # Every prevalence file must contain the same patients, so CV AUCs are
    # comparable across levels and only the feature set changes.
    run_set = set(df_level[RUN_COL])
    if reference_run_set is None:
        reference_run_set = run_set
        reference_run_set_threshold = threshold
    elif run_set != reference_run_set:
        only_here = run_set - reference_run_set
        only_ref = reference_run_set - run_set
        raise ValueError(
            f"Prevalence >= {pct} file has a DIFFERENT patient (Run) set than the "
            f"prevalence >= {reference_run_set_threshold:.0%} file. Run(s) only in this file: "
            f"{sorted(only_here)[:10]}. Run(s) only in the reference file: {sorted(only_ref)[:10]}."
        )

    print(f"  Running per-cohort Mann-Whitney U")
    results = compute_direction_pvalues(df_level, feature_cols_level)

    # Count how many markers each (p-threshold, min-cohorts) combination selects,
    # and cache the marker lists for scoring below.
    grid = pd.DataFrame(index=P_THRESHOLD_CANDIDATES, columns=MIN_CONSISTENT_COHORTS_CANDIDATES, dtype=int)
    combo_cache = {}
    for p_thresh in P_THRESHOLD_CANDIDATES:
        for min_cohorts in MIN_CONSISTENT_COHORTS_CANDIDATES:
            markers, direction_counts = markers_from_results(results, p_thresh, min_cohorts)
            grid.loc[p_thresh, min_cohorts] = len(markers)
            combo_cache[(p_thresh, min_cohorts)] = (markers, direction_counts)
    grid.index.name = "P_THRESHOLD"
    grid.columns.name = "MIN_CONSISTENT_COHORTS"
    print(f"  Survivor count grid (prevalence >= {pct}):")
    for line in grid.to_string().splitlines():
        print(f"    {line}")

    viable_combos = [
        (p, m) for p in P_THRESHOLD_CANDIDATES for m in MIN_CONSISTENT_COHORTS_CANDIDATES
        if grid.loc[p, m] >= MIN_SURVIVORS_FOR_VIABLE_COMBO
    ]
    print(f"  {len(viable_combos)} viable combos (>= {MIN_SURVIVORS_FOR_VIABLE_COMBO} survivors) at this level.")
    if not viable_combos:
        print(f"  Skipping prevalence >= {pct}: no combo reaches {MIN_SURVIVORS_FOR_VIABLE_COMBO} survivors.\n")
        continue

    y_level = (df_level[CLASS_COL] == "R").astype(int).to_numpy()  # responder = 1

    for (p_thresh, min_cohorts) in viable_combos:
        markers, direction_counts = combo_cache[(p_thresh, min_cohorts)]
        X_level = df_level[markers].to_numpy(dtype=float)
        estimator = build_estimator(MODEL_TO_RUN)
        scores = cross_val_score(estimator, X_level, y_level, cv=KFOLD, scoring="roc_auc", n_jobs=-1)
        auc = scores.mean()
        print(f"    P_THRESHOLD={p_thresh}, MIN_CONSISTENT_COHORTS={min_cohorts}: "
              f"{len(markers)} markers, CV AUC={auc:.4f} (default hyperparameters)")
        stage_a_records.append({
            "threshold": threshold,
            "p_thresh": p_thresh,
            "min_cohorts": min_cohorts,
            "markers": markers,
            "direction_counts": direction_counts,
            "cv_auc": auc,
        })
    print()

if not stage_a_records:
    raise ValueError(
        "No prevalence level / threshold-combo produced a viable marker set (>= "
        f"{MIN_SURVIVORS_FOR_VIABLE_COMBO} survivors) - cannot proceed."
    )

# Pick the configuration with the highest mean CV AUC.
best_a = max(stage_a_records, key=lambda r: r["cv_auc"])
print("Stage A winner")
print(f"  Prevalence >= {best_a['threshold']:.0%}, P_THRESHOLD={best_a['p_thresh']}, "
      f"MIN_CONSISTENT_COHORTS={best_a['min_cohorts']}: {len(best_a['markers'])} markers, "
      f"CV AUC={best_a['cv_auc']:.4f} (default hyperparameters)\n")

marker_features = best_a["markers"]
direction_counts = best_a["direction_counts"]
winning_threshold = best_a["threshold"]
winning_p_thresh = best_a["p_thresh"]
winning_min_cohorts = best_a["min_cohorts"]

# "R > NR" means enriched in responders, "NR > R" enriched in non-responders.
print("Winning config's marker features:")
for feat in marker_features:
    c = direction_counts[feat]
    dom_dir = "R > NR" if c["up"] >= c["down"] else "NR > R"
    print(f"  {feat}  (direction: {dom_dir}, up={c['up']}, down={c['down']})")
print()


# Reload the winning prevalence file to get the full discovery data for Stage B.
winning_df, winning_feature_cols_level = load_and_validate_discovery_file(
    PREVALENCE_FILE_PATHS[winning_threshold], "Stage A winner (reload)"
)


# Top-ranked biomarkers for reporting. min_cohorts is fixed at 3 here,
# regardless of the Stage A winner's setting.
winning_level_results = compute_direction_pvalues(winning_df, winning_feature_cols_level)
top_biomarkers = top_biomarkers_overall(
    winning_level_results, p_threshold=winning_p_thresh, min_cohorts=3, top_n=10
)
print_top_biomarkers(top_biomarkers, winning_p_thresh, min_cohorts=3)


## Stage B: hyperparameter grid search for the Stage A winner's marker set
# Exhaustive search, scored by mean 10-fold CV AUC on the pooled discovery cohorts.
print(f"Stage B: hyperparameter grid search for {model_cfg['display_name']} on Stage A winner\n")
X = winning_df[marker_features].to_numpy(dtype=float)
y = (winning_df[CLASS_COL] == "R").astype(int).to_numpy()  # responder = 1
print(f"  Discovery data for Stage B: {X.shape[0]} patients, {X.shape[1]} features.")

param_grid = model_cfg["param_grid"]
base_estimator = build_estimator(MODEL_TO_RUN)


# Total number of combinations for the log message
n_combos = 1
for v in param_grid.values():
     n_combos *= len(v)
print(f"  Using exhaustive GridSearchCV ({n_combos:,} combinations x {NUM_FOLDS}-fold CV "
        f"= {n_combos * NUM_FOLDS:,} fits)...")
search = GridSearchCV(base_estimator, param_grid, scoring="roc_auc", cv=KFOLD, n_jobs=-1)

search.fit(X, y)
best_params = search.best_params_
best_cv_auc = search.best_score_
final_estimator = search.best_estimator_  # already refit on ALL of X, y (GridSearchCV default refit=True)

print(f"\n  Best hyperparameters: {best_params}")
print(f"  Best CV AUC: {best_cv_auc:.4f}\n")


## Stage C: evaluate on Routy external validation set
def evaluate_external(test_csv_path, label):
    """Score the final model on an external cohort and report AUC.

    The external data play no part in marker selection or tuning. Only the Stage A
    marker panel is used, and any marker missing from the file is zero-filled
    (treated as "not detected"). The outcome column must be named "Class" and
    coded "R"/"NR".
    """
    print(f"Evaluating on {label}")
    test_df_full = pd.read_csv(test_csv_path)

    if test_df_full.columns.duplicated().any():
        raise ValueError(f"{label}: duplicate column name(s): "
                          f"{test_df_full.columns[test_df_full.columns.duplicated()].tolist()}")

    if CLASS_COL not in test_df_full.columns:
        raise ValueError(f"{label}: no '{CLASS_COL}' column found. "
                          f"Columns present: {list(test_df_full.columns)}")

    missing_markers = [f for f in marker_features if f not in test_df_full.columns]
    present_markers = [f for f in marker_features if f in test_df_full.columns]

    if missing_markers:
        frac_missing = len(missing_markers) / len(marker_features)
        print(f"  NOTE: {len(missing_markers)}/{len(marker_features)} marker feature(s) "
              f"({frac_missing:.1%}) absent from {label} - zero-filling (treated as "
              f"'not detected in any sample'):")
        for f in missing_markers:
            print(f"    {f}")
        if frac_missing > MAX_ZEROFILL_FRACTION_WARN:
            print(f"WARNING: {frac_missing:.1%} of the marker panel was zero-filled "
                  f"for {label} - exceeds the {MAX_ZEROFILL_FRACTION_WARN:.0%} caution threshold.")
        print()

    # Keep only the marker columns, in the same order as during training.
    test_sub = test_df_full[present_markers + [CLASS_COL]].copy()
    for f in missing_markers:
        test_sub[f] = 0.0
    test_sub = test_sub[marker_features + [CLASS_COL]]

    unexpected = set(test_sub[CLASS_COL].dropna().unique()) - {"R", "NR"}
    if unexpected:
        raise ValueError(f"{label}: unexpected {CLASS_COL} value(s): {unexpected}")
    if test_sub[CLASS_COL].isna().any():
        raise ValueError(f"{label}: {test_sub[CLASS_COL].isna().sum()} rows have a missing "
                          f"{CLASS_COL} value - cannot evaluate those rows.")

    X_ext = test_sub[marker_features].to_numpy(dtype=float)
    y_ext = (test_sub[CLASS_COL] == "R").astype(int).to_numpy()  # responder = 1

    y_score_ext = get_scores(final_estimator, X_ext)
    ext_metrics = compute_metrics(y_ext, y_score_ext)

    print(f"{label}: n={len(y_ext)}, AUC={ext_metrics['auc']:.4f}\n")

    roc_plot_path = os.path.join(r"path/to/output_figures", f"{MODEL_TO_RUN}_{label}_roc_curve.png")  # placeholder; folder must already exist
    plot_roc_curve(y_ext, y_score_ext, ext_metrics["auc"], label, roc_plot_path)

    # Optional: permutation test on the external AUC (off by default).
    if RUN_PERMUTATION_TEST:
        print(f"Label permutation test on {label} ({N_PERMUTATIONS} permutations)")
        observed_auc, null_aucs, p_value = run_permutation_test(
            y_ext, y_score_ext, N_PERMUTATIONS, PERMUTATION_SEED
        )
        print(f"  Observed {label} AUC: {observed_auc:.4f}")
        print(f"  Null distribution:    mean={null_aucs.mean():.4f}, std={null_aucs.std():.4f}")
        floor_p = 1 / (N_PERMUTATIONS + 1)
        if p_value <= floor_p:
            print(f"  Permutation p-value: p < {floor_p:.4f} (one-sided, {N_PERMUTATIONS} permutations, "
                  f"hit the floor - p < {floor_p:.3f}")
        else:
            print(f"  Permutation p-value: {p_value:.4f} (one-sided, {N_PERMUTATIONS} permutations)")

        perm_plot_path = os.path.join(r"path/to/output_figures", f"{MODEL_TO_RUN}_{label}_permutation_test.png")  # placeholder; folder must already exist
        plot_permutation_distribution(null_aucs, observed_auc, p_value, N_PERMUTATIONS, perm_plot_path)
        print()

    return ext_metrics


routy_metrics = evaluate_external(ROUTY_TEST_PATH, "Routy")

print("FINAL SUMMARY")
print(f"Model: {model_cfg['display_name']} ({MODEL_TO_RUN})")
print(f"Winning prevalence filter: >= {winning_threshold:.0%}")
print(f"Winning P_THRESHOLD={winning_p_thresh}, MIN_CONSISTENT_COHORTS={winning_min_cohorts}")
print(f"Marker features used: {len(marker_features)}")
print(f"Winning hyperparameters: {best_params}")
print(f"Discovery CV AUC: {best_cv_auc:.4f}")
print(f"Routy external test: AUC={routy_metrics['auc']:.4f}")
