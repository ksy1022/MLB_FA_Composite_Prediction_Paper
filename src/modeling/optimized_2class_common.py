"""Shared utilities for the revised MLB FA binary-classification experiment.

This module intentionally contains NO hard-coded role-specific High ratios.
The role-specific percentile cutoff is selected from development data by
`run_optimized_2class.py` and then locked before the 2020-2021 test set is used.

Expected repository layout
--------------------------
<repo>/
    data/processed_v2/{hitter,sp,rp}_cohort_v2.csv
    src/modeling/optimized_2class_common.py

Temporal split
--------------
Train      : 2012-2017
Validation : 2018-2019
Test       : 2020-2021
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed_v2"
OUTPUT = ROOT / "outputs" / "optimized_2class"
RANDOM_STATE = 42

ROLE_CONFIG: dict[str, dict[str, Any]] = {
    "hitter": {
        "war": "post_WAR_3y_sum",
        "playtime": "post_PA_3y",
        "features": [
            "Pos_group",
            "age_at_contract",
            "pre_PA_3y",
            "pre_WAR_3y_avg",
            "pre_WAR_3y_std",
            "pre_OPSplus_3y_avg",
            "pre_BB_rate_3y_avg",
            "pre_K_rate_3y_avg",
            "AAV",
            "Yrs",
        ],
    },
    "sp": {
        "war": "post_WAR_3y_sum",
        "playtime": "post_IP_3y",
        "features": [
            "age_at_contract",
            "pre_IP_3y",
            "pre_WAR_3y_avg",
            "pre_WAR_3y_std",
            "pre_ERA_3y_avg",
            "pre_FIP_3y_avg",
            "pre_KBB_3y_avg",
            "pre_GS_3y_sum",
            "pre_G_3y_sum",
            "AAV",
            "Yrs",
        ],
    },
    "rp": {
        "war": "post_WAR_3y_sum",
        "playtime": "post_IP_3y",
        "features": [
            "age_at_contract",
            "pre_IP_3y",
            "pre_WAR_3y_avg",
            "pre_WAR_3y_std",
            "pre_ERA_3y_avg",
            "pre_FIP_3y_avg",
            "pre_KBB_3y_avg",
            "pre_GS_3y_sum",
            "pre_G_3y_sum",
            "AAV",
            "Yrs",
        ],
    },
}

ROLE_LABELS = {
    "hitter": "Hitter",
    "sp": "Starting Pitcher",
    "rp": "Relief Pitcher",
}

MODEL_COMPLEXITY = {"logistic": 0, "rf": 1, "xgb": 2}


def temporal_split(df: pd.DataFrame) -> pd.Series:
    """Return train/valid/test labels using FA contract year only."""
    year = pd.to_numeric(df["fa_year"], errors="coerce")
    split = pd.Series(index=df.index, dtype="object")
    split[year.between(2012, 2017)] = "train"
    split[year.between(2018, 2019)] = "valid"
    split[year.between(2020, 2021)] = "test"
    return split


def load_role_frame(role: str) -> pd.DataFrame:
    path = DATA / f"{role}_cohort_v2.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cohort: {path}")
    df = pd.read_csv(path, dtype={"player_key": str})
    df = df.reset_index(drop=True)
    df["contract_row_id"] = np.arange(len(df), dtype=int)
    df["split"] = temporal_split(df)
    df = df[df["split"].notna()].copy()
    if df.empty:
        raise ValueError(f"{role}: no rows in 2012-2021")
    return df


def prepare_scores(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Create leakage-safe standardized WAR, utilization, and equal-weight Composite.

    The imputation median, mean, and standard deviation are estimated ONLY from
    the 2012-2017 training partition and then applied unchanged to validation
    and test rows.
    """
    out = df.copy()
    train_mask = out["split"].eq("train")
    params: dict[str, float] = {}

    for column, short in ((cfg["war"], "war"), (cfg["playtime"], "playtime")):
        values = pd.to_numeric(out[column], errors="coerce")
        train_values = values.loc[train_mask]
        median = float(train_values.median())
        filled_train = train_values.fillna(median)
        mean = float(filled_train.mean())
        std = float(filled_train.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Training SD is invalid for {column}: {std}")
        out[column] = values
        out[f"z_{short}"] = (values.fillna(median) - mean) / std
        params.update(
            {
                f"{short}_median": median,
                f"{short}_mean": mean,
                f"{short}_std": std,
            }
        )

    # Explicitly use the paper's equal-weight notation.
    out["composite_score"] = 0.5 * out["z_war"] + 0.5 * out["z_playtime"]
    return out, params


def add_binary_labels(
    scored: pd.DataFrame,
    cfg: dict[str, Any],
    top_ratio: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Create Composite and WAR-only High labels using the same top proportion.

    The numerical cut point for each outcome is estimated from the training
    distribution only. The same cut point is then applied to validation/test.
    """
    if not (0.0 < top_ratio < 0.5):
        raise ValueError(f"top_ratio must be between 0 and 0.5, got {top_ratio}")

    out = scored.copy()
    train_mask = out["split"].eq("train")
    params: dict[str, float] = {"top_ratio": float(top_ratio)}

    score_map = {
        "composite": "composite_score",
        "war": cfg["war"],
    }
    for outcome, score_col in score_map.items():
        train_score = pd.to_numeric(out.loc[train_mask, score_col], errors="coerce")
        cut = float(train_score.quantile(1.0 - top_ratio))
        out[f"{outcome}_2class"] = (
            pd.to_numeric(out[score_col], errors="coerce").ge(cut).astype(int)
        )
        params[f"{outcome}_cut"] = cut
    return out, params


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def candidate_models(y_train: pd.Series) -> dict[str, object]:
    """Return the same three candidate model families used in the paper."""
    y_train = pd.Series(y_train).astype(int)
    result: dict[str, object] = {
        "logistic": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=RANDOM_STATE,
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }
    if XGBClassifier is not None:
        positives = int(y_train.sum())
        negatives = int(len(y_train) - positives)
        result["xgb"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=1,
            scale_pos_weight=(negatives / positives if positives else 1.0),
            eval_metric="logloss",
        )
    return result


def fit_pipeline(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    estimator: object,
) -> Pipeline:
    pipe = Pipeline(
        [
            ("preprocessor", make_preprocessor(train[features])),
            ("model", clone(estimator)),
        ]
    )
    pipe.fit(train[features], train[target])
    return pipe


def _safe_roc_auc(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y_true)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, probability))


def _safe_pr_auc(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y_true)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, probability))


def binary_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probability, dtype=float)
    pred = (p >= threshold).astype(int)
    eps = 1e-7
    p_clip = np.clip(p, eps, 1 - eps)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y, p),
        "pr_auc": _safe_pr_auc(y, p),
        "log_loss": float(log_loss(y, p_clip, labels=[0, 1])),
    }
    return metrics, pred


def select_probability_threshold(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    minimum: float = 0.05,
    maximum: float = 0.95,
    step: float = 0.01,
) -> tuple[float, pd.DataFrame]:
    """Choose the decision threshold on validation data only.

    Primary criterion: High-class F1
    Tie-breakers: balanced accuracy, recall, closeness to 0.50.
    """
    rows: list[dict[str, float]] = []
    for threshold in np.arange(minimum, maximum + step / 2, step):
        metrics, _ = binary_metrics(y_true, probability, float(threshold))
        rows.append(metrics)
    table = pd.DataFrame(rows)
    table["distance_from_0_5"] = (table["threshold"] - 0.5).abs()
    best = table.sort_values(
        ["f1", "balanced_accuracy", "recall", "distance_from_0_5"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]
    return float(best["threshold"]), table


def choose_best_validation_row(rows: pd.DataFrame) -> pd.Series:
    """Deterministically select a model/cutoff configuration on validation data.

    Selection order:
      1) High-class F1
      2) balanced accuracy
      3) PR-AUC
      4) recall
      5) more validation positives (stability tie-break)
      6) simpler model family when still exactly tied
    """
    if rows.empty:
        raise ValueError("No validation rows to select from")
    table = rows.copy()
    table["model_complexity"] = table["model"].map(MODEL_COMPLEXITY).fillna(99)
    sort_cols = [
        "f1",
        "balanced_accuracy",
        "pr_auc",
        "recall",
        "valid_positive_count",
        "model_complexity",
    ]
    # NaN PR-AUC should rank last.
    table["pr_auc"] = table["pr_auc"].fillna(-np.inf)
    best = table.sort_values(
        sort_cols,
        ascending=[False, False, False, False, False, True],
        kind="mergesort",
    ).iloc[0]
    return best
