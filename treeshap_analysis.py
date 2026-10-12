"""
TreeSHAP feature-importance analysis for the final biomarker model.
Code accompanying the manuscript submitted to Clinical Lung Cancer.

Trains a Random Forest or XGBoost classifier on the full discovery set
(restricted to the selected biomarkers), explains its predictions on the
external validation cohort with TreeSHAP, and reports the top biomarkers by
mean |SHAP| value.

Usage: set MODEL_TYPE and the file paths in the CONFIG section, then run.

Inputs:  a discovery CSV (features plus a Class column), a validation CSV with
         the same feature columns (Class not required), and a text file
         listing one biomarker per line.
Outputs: a CSV of the top-N biomarkers, a bar plot, and a beeswarm plot (PNG).

Dependencies: numpy v2.2.6, pandas v2.3.1, scipy v1.15.3, scikit-learn v1.7.2, xgboost v3.2.0, matplotlib v3.10.9, shap v0.49.0
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def _patch_shap_xgboost_base_score_bug():
    """
    Workaround for a shap/xgboost version incompatibility.

    xgboost >= 3.0 saves the model's base_score as a bracketed string (e.g.
    "[5.146199E-1]"). shap versions before ~0.52 call float() on it directly,
    which fails when building a TreeExplainer for an XGBoost model. This patch
    strips the brackets from the decoded model before shap parses it. Binary
    classification has a single base_score, so the value itself is unchanged.
    It has no effect if base_score is already a plain number.
    """
    try:
        import shap.explainers._tree as _shap_tree_mod
    except ImportError:
        return

    # Only patch once
    if getattr(_shap_tree_mod, "_base_score_bracket_patch_applied", False):
        return

    _original_decode = _shap_tree_mod.decode_ubjson_buffer

    def _patched_decode(fd):
        jmodel = _original_decode(fd)
        try:
            lmp = jmodel["learner"]["learner_model_param"]
            bs = lmp["base_score"]
            if isinstance(bs, str) and bs.strip().startswith("["):
                lmp["base_score"] = bs.strip().strip("[]")
        except (KeyError, TypeError):
            pass
        return jmodel

    _shap_tree_mod.decode_ubjson_buffer = _patched_decode
    _shap_tree_mod._base_score_bracket_patch_applied = True


_patch_shap_xgboost_base_score_bug()

# CONFIG
MODEL_TYPE = "xgboost"  # "xgboost" or "rf"

DISCOVERY_DATA_PATH = r"path/to/discovery_data.csv"  # placeholder - set to your local path
VALIDATION_DATA_PATH = r"path/to/validation_data.csv"  # placeholder
BIOMARKER_LIST_PATH = r"path/to/biomarker_list.txt"  # placeholder

TARGET_COLUMN = "Class"    # label column in the discovery file (not needed in the validation file)
                            # "R"/"NR" are encoded alphabetically by LabelEncoder (NR = 0, R = 1),
                            # so a positive SHAP value pushes a prediction toward R and a
                            # negative one toward NR.

RANDOM_STATE = 1            # same seed as the classification pipeline

OUTPUT_DIR = r"path/to/output_figures"  # placeholder
TOP_N = 10                  # number of top biomarkers to report and plot

# Final model hyperparameters, as selected in the classification pipeline
RF_PARAMS = {
    "criterion": "entropy",
    "max_depth": 19,
    "max_features": "log2",
    "min_samples_leaf": 3,
    "n_estimators": 500,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "colsample_bytree": 1.0,
    "learning_rate": 0.075,
    "max_depth": 9,
    "n_estimators": 100,
    "subsample": 1.0,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "tree_method": "hist",
}


# Taxonomic rank prefixes used in the feature names (MetaPhlAn-style),
# e.g. "s__Blautia_wexlerae". Only genus and species names are italicized,
# following standard microbiology convention.
_TAXON_RANK_PREFIXES = {
    "s__": "species",
    "g__": "genus",
    "f__": "family",
    "o__": "order",
    "c__": "class",
    "p__": "phylum",
    "k__": "kingdom",
}
_ITALIC_RANKS = {"species", "genus"}


def clean_taxon_label(raw_name):
    """
    Convert a raw feature name (e.g. "s__Blautia_wexlerae", optionally with a
    "|"-separated lineage) into a display name for figures.

    Returns (name, italicize), where italicize is True for genus and species
    names only.
    """
    token = raw_name.split("|")[-1].strip() if "|" in raw_name else raw_name.strip()
    for prefix, rank in _TAXON_RANK_PREFIXES.items():
        if token.startswith(prefix):
            name = token[len(prefix):].replace("_", " ").strip()
            return name, rank in _ITALIC_RANKS
    # Unrecognized format: just remove underscores, no italics
    return token.replace("_", " "), False


def clean_numeric_columns(df, cols):
    """
    Convert string-valued feature columns to numeric.

    Some exported tables (e.g. after batch correction in R) store numbers as
    strings, sometimes wrapped in brackets like "[5.146199E-1]". Brackets,
    quotes and whitespace are stripped and the values coerced to numeric.
    Anything that is still unparseable becomes NaN and triggers a warning.
    """
    cleaned = df.copy()
    n_new_nan = 0
    for c in cols:
        if pd.api.types.is_numeric_dtype(cleaned[c]):
            continue
        before_na = cleaned[c].isna().sum()
        stripped = cleaned[c].astype(str).str.strip().str.strip("[]").str.strip("'\"")
        numeric = pd.to_numeric(stripped, errors="coerce")
        n_new_nan += numeric.isna().sum() - before_na
        cleaned[c] = numeric
    if n_new_nan > 0:
        warnings.warn(
            f"{n_new_nan} value(s) could not be parsed as numeric after "
            "stripping brackets/quotes and were set to NaN. Inspect these "
            "columns - this may indicate a formatting issue beyond simple "
            "bracket-wrapping."
        )
    return cleaned


def load_biomarker_list(path):
    """Read biomarker names from a text file, one per line, skipping blank lines."""
    with open(path) as f:
        biomarkers = [line.strip() for line in f if line.strip()]
    if not biomarkers:
        raise ValueError(f"No biomarkers found in {path}")
    return biomarkers


def load_and_restrict_data(data_path, biomarker_path, target_col):
    """
    Load the discovery table, keeping only the selected biomarkers and the
    target column. Raises an error if any biomarker or the target is missing.
    """
    df = pd.read_csv(data_path)
    biomarkers = load_biomarker_list(biomarker_path)

    missing = [b for b in biomarkers if b not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} biomarker(s) from {biomarker_path} not found in "
            f"{data_path}: {missing}"
        )
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {data_path}")

    df = df[biomarkers + [target_col]].copy()
    df = clean_numeric_columns(df, biomarkers)

    # Missing values are flagged but not imputed
    n_missing = df[biomarkers].isna().sum().sum()
    if n_missing > 0:
        warnings.warn(
            f"{n_missing} missing values found in biomarker columns. "
        )

    X = df[biomarkers]
    y_raw = df[target_col]
    return X, y_raw, biomarkers


def load_and_restrict_features(data_path, biomarkers):
    """
    Same as load_and_restrict_data, but for the validation set. No target
    column is needed because TreeSHAP only requires the features to explain
    the trained model's predictions.
    """
    df = pd.read_csv(data_path)

    missing = [b for b in biomarkers if b not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} biomarker(s) not found in {data_path}: {missing}. "
            "The validation set must contain the exact same feature columns "
            "(same names) used to train the model."
        )

    X = clean_numeric_columns(df[biomarkers].copy(), biomarkers)

    n_missing = X.isna().sum().sum()
    if n_missing > 0:
        warnings.warn(
            f"{n_missing} missing values found in validation biomarker columns. "
        )

    return X


def encode_target(y_raw):
    """
    Encode string labels as integers (alphabetical: NR = 0, R = 1) and print
    the mapping. Numeric labels are assumed to already be coded 0/1 with
    1 = responder.
    """
    if not pd.api.types.is_numeric_dtype(y_raw) or pd.api.types.is_bool_dtype(y_raw):
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Target classes encoded as: {mapping}")
        return y, le
    # Already numeric
    return y_raw.to_numpy(), None


def train_model(X, y, model_type):
    """Fit the selected model (Random Forest or XGBoost) on the full discovery set."""
    if model_type == "rf":
        model = RandomForestClassifier(**RF_PARAMS)
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(**XGB_PARAMS)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")
    model.fit(X, y)
    return model


def get_positive_class_shap(shap_values):
    """
    Return signed SHAP values for the positive class (R) as an
    (n_samples, n_features) array.

    Random Forest explanations come back as 3D (samples, features, classes)
    because the model outputs class probabilities. XGBoost binary:logistic
    explanations are 2D because SHAP explains the log-odds margin, which is a
    single score. Binary classification only.
    """
    values = shap_values.values

    if values.ndim == 3:
        n_classes = values.shape[2]
        if n_classes != 2:
            raise ValueError(
                f"Detected {n_classes} classes — this script assumes binary "
                "classification."
            )
        return values[:, :, 1]
    elif values.ndim == 2:
        return values
    else:
        raise ValueError(f"Unexpected SHAP values shape: {values.shape}")


def summarize_top_features(shap_matrix, X, feature_names, top_n):
    """
    Rank features by mean |SHAP| and assign each a direction of effect.

    Direction is the sign of the Spearman correlation between a feature's
    values and its SHAP values, not the sign of the mean SHAP value, which
    depends on the class balance of the validation set. For example, a feature
    that pushes predictions toward R can still have a negative mean SHAP if
    most samples are NR.

    "+" = higher values push toward R, "-" = toward NR, "?" = undetermined.
    Returns a DataFrame of the top features and their column indices.
    """
    from scipy.stats import spearmanr

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1][:top_n]

    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    rows = []
    for rank, idx in enumerate(order, start=1):
        feature_vals = X_arr[:, idx]
        shap_vals = shap_matrix[:, idx]
        # Correlation is undefined if the feature or its SHAP values are constant
        if np.std(feature_vals) == 0 or np.std(shap_vals) == 0:
            corr = np.nan
            direction = "?"
            warnings.warn(
                f"Feature '{feature_names[idx]}' has zero variance in either "
                "its values or SHAP values on this validation set - direction "
                "could not be determined."
            )
        else:
            corr, _ = spearmanr(feature_vals, shap_vals)
            direction = "+" if corr >= 0 else "-"

        display_name, italic = clean_taxon_label(feature_names[idx])
        rows.append(
            {
                "rank": rank,
                "feature": feature_names[idx],
                "taxon_label": display_name,
                "mean_abs_shap": mean_abs_shap[idx],
                "shap_value_correlation": corr,
                "direction": direction,
            }
        )
    return pd.DataFrame(rows), order


def make_bar_plot(top_df, output_path, title, class_labels=None):
    """
    Horizontal bar chart of mean |SHAP| for the top features, colored by
    direction of effect, with genus/species names italicized.

    class_labels is an optional (negative, positive) tuple such as ("NR", "R")
    used for the legend.
    """
    POS_COLOR = "#008080"  # positive-encoded class (R)
    NEG_COLOR = "#FF7F50"  # negative-encoded class (NR)
    UNKNOWN_COLOR = "#B0B0B0"  # direction undetermined (zero variance)

    # Reverse so rank 1 is plotted at the top
    df = top_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(df))
    colors = [
        POS_COLOR if d == "+" else NEG_COLOR if d == "-" else UNKNOWN_COLOR
        for d in df["direction"]
    ]

    fig_height = max(3, 0.5 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bars = ax.barh(y_pos, df["mean_abs_shap"], color=colors, edgecolor="black", linewidth=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["taxon_label"], fontsize=11)

    # Italicize genus/species labels only
    for label, is_italic in zip(ax.get_yticklabels(), df["feature"].map(
        lambda f: clean_taxon_label(f)[1]
    )):
        if is_italic:
            label.set_fontstyle("italic")

    # Value labels at the end of each bar
    max_val = df["mean_abs_shap"].max()
    for bar, val in zip(bars, df["mean_abs_shap"]):
        ax.text(
            bar.get_width() + max_val * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Mean |SHAP value| (impact on model output)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, max_val * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if class_labels:
        neg_name, pos_name = class_labels
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=POS_COLOR, label=f"Associated with {pos_name}"),
            plt.Rectangle((0, 0), 1, 1, color=NEG_COLOR, label=f"Associated with {neg_name}"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_beeswarm_plot(shap_matrix, X, feature_names, output_path, title):
    """SHAP beeswarm plot with display-formatted taxon names (genus/species italicized)."""
    display_names = []
    italic_flags = []
    for f in feature_names:
        name, italic = clean_taxon_label(f)
        display_names.append(name)
        italic_flags.append(italic)
    italic_lookup = dict(zip(display_names, italic_flags))

    fig_height = max(4, 0.45 * len(feature_names) + 1.5)
    plt.figure(figsize=(10, fig_height))
    shap.summary_plot(
        shap_matrix, X, feature_names=display_names, show=False, plot_size=None
    )

    ax = plt.gca()
    for label in ax.get_yticklabels():
        if italic_lookup.get(label.get_text(), False):
            label.set_fontstyle("italic")

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_plots(shap_matrix, X, feature_names, top_df, top_idx, output_dir,
                model_type, n_samples, class_labels=None):
    """Save the bar plot and beeswarm plot for the top features."""
    os.makedirs(output_dir, exist_ok=True)

    # Restrict to the top features
    top_feature_names = [feature_names[i] for i in top_idx]
    shap_top = shap_matrix[:, top_idx]
    X_top = X.iloc[:, top_idx]

    top_n = len(top_idx)
    model_label = "Random Forest" if model_type == "rf" else "XGBoost"
    bar_title = (
        f"Top {top_n} Biomarkers by TreeSHAP Importance\n"
        f"{model_label} Model, Validation Set (n={n_samples})"
    )
    beeswarm_title = (
        f"TreeSHAP Summary - Top {top_n} Biomarkers\n"
        f"{model_label} Model, Validation Set (n={n_samples})"
    )

    make_bar_plot(
        top_df,
        os.path.join(output_dir, f"top{top_n}_shap_bar.png"),
        bar_title,
        class_labels=class_labels,
    )

    make_beeswarm_plot(
        shap_top, X_top, top_feature_names,
        os.path.join(output_dir, f"top{top_n}_shap_beeswarm.png"),
        beeswarm_title,
    )

    print(f"Saved plots to {output_dir}/")


def main():
    # Train on the full discovery set, restricted to the selected biomarkers
    X_train, y_raw, biomarkers = load_and_restrict_data(
        DISCOVERY_DATA_PATH, BIOMARKER_LIST_PATH, TARGET_COLUMN
    )
    y_train, label_encoder = encode_target(y_raw)

    model = train_model(X_train, y_train, MODEL_TYPE)

    # Load the external validation set with the same biomarkers
    X_val = load_and_restrict_features(VALIDATION_DATA_PATH, biomarkers)

    # Explain the model's predictions on the validation set with TreeSHAP
    # (the reported results are based on this set)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_val)

    shap_matrix = get_positive_class_shap(shap_values)

    top_df, top_idx = summarize_top_features(shap_matrix, X_val, biomarkers, TOP_N)

    print(f"\nTop {TOP_N} biomarkers by mean |SHAP| (evaluated on validation set):")
    print(top_df.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    top_df.to_csv(os.path.join(OUTPUT_DIR, f"top{TOP_N}_biomarkers.csv"), index=False)

    class_labels = tuple(label_encoder.classes_) if label_encoder is not None else None
    make_plots(
        shap_matrix, X_val, biomarkers, top_df, top_idx, OUTPUT_DIR,
        model_type=MODEL_TYPE, n_samples=X_val.shape[0], class_labels=class_labels,
    )


if __name__ == "__main__":
    main()
