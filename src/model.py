"""SVM training, evaluation, and visualisation utilities for HOG-based solar panel detection."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm


def _build_pipeline(kernel, C, gamma, class_weight="balanced", probability=True):
    """Internal helper: StandardScaler + SVC pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=kernel, C=C, gamma=gamma,
                    class_weight=class_weight, probability=probability,
                    random_state=42)),
    ])


def make_group_cv(n_splits=5):
    """Return a StratifiedGroupKFold splitter with shuffle and fixed seed."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)


def tune_svm(X, y, groups, cv, scoring="f1"):
    """Joint (kernel, C, gamma) grid search; returns (results_df, best_estimator, best_params_dict)."""
    linear_pipe = _build_pipeline(kernel="linear", C=1.0, gamma="scale")
    linear_grid = GridSearchCV(
        linear_pipe,
        param_grid={"svc__C": np.logspace(-2, 2, 5), "svc__class_weight": ["balanced"]},
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    linear_grid.fit(X, y, groups=groups)

    rbf_pipe = _build_pipeline(kernel="rbf", C=1.0, gamma="scale")
    rbf_grid = GridSearchCV(
        rbf_pipe,
        param_grid={
            "svc__C": np.logspace(-2, 2, 5),
            "svc__gamma": np.logspace(-4, 1, 6),
            "svc__class_weight": ["balanced"],
        },
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    rbf_grid.fit(X, y, groups=groups)

    def _rows(grid, kernel_name):
        res = grid.cv_results_
        out = []
        for i in range(len(res["mean_test_score"])):
            params = res["params"][i]
            out.append({
                "kernel":          kernel_name,
                "C":               params["svc__C"],
                "gamma":           params.get("svc__gamma", np.nan),
                "mean_test_score": res["mean_test_score"][i],
                "std_test_score":  res["std_test_score"][i],
                "mean_fit_time":   res["mean_fit_time"][i],
                "rank_test_score": res["rank_test_score"][i],
            })
        return out

    results_df = (
        pd.DataFrame(_rows(linear_grid, "linear") + _rows(rbf_grid, "rbf"))
        .sort_values("mean_test_score", ascending=False)
        .reset_index(drop=True)
    )

    if linear_grid.best_score_ >= rbf_grid.best_score_:
        best_estimator = linear_grid.best_estimator_
        best_params = {
            k.replace("svc__", ""): v
            for k, v in linear_grid.best_params_.items()
            if k != "svc__class_weight"
        }
        best_params["kernel"] = "linear"
    else:
        best_estimator = rbf_grid.best_estimator_
        best_params = {
            k.replace("svc__", ""): v
            for k, v in rbf_grid.best_params_.items()
            if k != "svc__class_weight"
        }
        best_params["kernel"] = "rbf"

    return results_df, best_estimator, best_params


def tune_threshold(model, X, y, groups, cv):
    """OOF threshold sweep on positive-class probabilities; returns (best_threshold, threshold_df)."""
    y_prob = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": t,
            "f1":        f1_score(y, y_pred, zero_division=0),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall":    recall_score(y, y_pred, zero_division=0),
        })
    df = pd.DataFrame(rows)
    best_threshold = float(df.loc[df["f1"].idxmax(), "threshold"])
    return best_threshold, df


def evaluate_on_holdout(model, X_test, y_test, threshold=0.5):
    """Evaluate on held-out test set; applies threshold to predict_proba to produce y_pred."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":         accuracy_score(y_test, y_pred),
        "precision":        precision_score(y_test, y_pred, zero_division=0),
        "recall":           recall_score(y_test, y_pred, zero_division=0),
        "f1":               f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":          roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred":           y_pred,
        "y_prob":           y_prob,
        "threshold":        threshold,
    }


def run_hog_ablation(npz_paths_with_meta, groups_per_dataset, best_params, cv):
    """Cross-validate each ablation .npz using best_params; returns DataFrame sorted by mean_f1."""
    from feature_extraction import load_feature_matrix  # local import keeps namespace clean

    kernel = best_params["kernel"]
    C      = best_params["C"]
    gamma  = best_params.get("gamma", "scale")

    rows = []
    for meta, groups in tqdm(
        zip(npz_paths_with_meta, groups_per_dataset),
        total=len(npz_paths_with_meta),
        desc="HOG ablation",
    ):
        X, y, *_ = load_feature_matrix(meta["path"])
        pipe = _build_pipeline(kernel=kernel, C=C, gamma=gamma,
                                class_weight="balanced", probability=True)
        cv_res = cross_validate(
            pipe, X, y, groups=groups, cv=cv,
            scoring=["accuracy", "roc_auc", "f1", "precision", "recall"],
            return_train_score=False,
            n_jobs=1,
        )
        rows.append({
            "label":           meta["label"],
            "pixels_per_cell": meta["pixels_per_cell"],
            "cells_per_block": meta["cells_per_block"],
            "orientations":    meta["orientations"],
            "n_samples":       X.shape[0],
            "n_features":      X.shape[1],
            "mean_f1":         cv_res["test_f1"].mean(),
            "std_f1":          cv_res["test_f1"].std(),
            "mean_roc_auc":    cv_res["test_roc_auc"].mean(),
            "std_roc_auc":     cv_res["test_roc_auc"].std(),
            "mean_accuracy":   cv_res["test_accuracy"].mean(),
            "std_accuracy":    cv_res["test_accuracy"].std(),
            "mean_precision":  cv_res["test_precision"].mean(),
            "std_precision":   cv_res["test_precision"].std(),
            "mean_recall":     cv_res["test_recall"].mean(),
            "std_recall":      cv_res["test_recall"].std(),
        })

    return pd.DataFrame(rows).sort_values("mean_f1", ascending=False).reset_index(drop=True)


def plot_roc_curve(y_true, y_prob, title="ROC Curve", ax=None):
    """Plot ROC curve on ax (creates figure if None)."""
    if ax is None:
        _, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random classifier")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax


def plot_precision_recall_curve(y_true, y_prob, title="Precision-Recall Curve", ax=None):
    """Plot Precision-Recall curve on ax (creates figure if None)."""
    if ax is None:
        _, ax = plt.subplots()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"Avg Precision = {ap:.4f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return ax


def plot_confusion_matrix(cm, title="Confusion Matrix", ax=None):
    """Plot confusion matrix as a seaborn heatmap on ax (creates figure if None)."""
    if ax is None:
        _, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=0.5, linecolor="grey")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_threshold_sweep(threshold_df, ax=None):
    """Plot F1, precision, and recall vs decision threshold from tune_threshold output."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    ax.plot(threshold_df["threshold"], threshold_df["f1"],        label="F1",        color="seagreen",  lw=2)
    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", color="steelblue", lw=2)
    ax.plot(threshold_df["threshold"], threshold_df["recall"],    label="Recall",    color="salmon",    lw=2)
    best = threshold_df.loc[threshold_df["f1"].idxmax()]
    ax.axvline(best["threshold"], color="grey", linestyle="--", lw=1.5,
               label=f"Best threshold = {best['threshold']:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("OOF Threshold Sweep")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    return ax
