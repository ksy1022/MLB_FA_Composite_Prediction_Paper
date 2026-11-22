"""Run a reproducible 2-class pilot on the leakage-safe V2 splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
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
INPUT = ROOT / "outputs" / "leakage_safe_splits"
OUTPUT = ROOT / "outputs" / "v2_2class"

FEATURES = {
    "hitter": [
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
    "sp": [
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
    "rp": [
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
}


def preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = frame.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [column for column in frame if column not in numeric_cols]
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
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def models(y_train: pd.Series) -> dict[str, object]:
    positive = int(y_train.sum())
    negative = len(y_train) - positive
    scale = negative / positive if positive else 1.0
    result: dict[str, object] = {
        "logistic": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000, random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        result["xgb"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    return result


def threshold_by_validation_f1(y_true: pd.Series, probability: np.ndarray) -> float:
    candidates = np.arange(0.10, 0.91, 0.01)
    scores = [f1_score(y_true, probability >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def metrics(
    dataset: str,
    model: str,
    split: str,
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    prediction = (probability >= threshold).astype(int)
    return {
        "dataset": dataset,
        "model": model,
        "split": split,
        "threshold": threshold,
        "n": len(y_true),
        "positive_rate": float(y_true.mean()),
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
    }


def run_dataset(name: str) -> list[dict[str, object]]:
    df = pd.read_csv(INPUT / f"{name}_labeled_splits.csv")
    features = [column for column in FEATURES[name] if column in df]
    train = df[df["split"].eq("train")]
    valid = df[df["split"].eq("valid")]
    test = df[df["split"].eq("test")]
    target = "success_2class"
    rows: list[dict[str, object]] = []
    for model_name, estimator in models(train[target]).items():
        pipeline = Pipeline(
            [("preprocessor", preprocessor(train[features])), ("model", estimator)]
        )
        pipeline.fit(train[features], train[target])
        valid_probability = pipeline.predict_proba(valid[features])[:, 1]
        threshold = threshold_by_validation_f1(valid[target], valid_probability)
        test_probability = pipeline.predict_proba(test[features])[:, 1]
        rows.append(
            metrics(name, model_name, "valid", valid[target], valid_probability, threshold)
        )
        rows.append(
            metrics(name, model_name, "test", test[target], test_probability, threshold)
        )
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for name in ("hitter", "sp", "rp"):
        rows.extend(run_dataset(name))
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "v2_2class_metrics.csv", index=False, encoding="utf-8-sig")
    print(
        result[result["split"].eq("test")]
        .sort_values(["dataset", "f1"], ascending=[True, False])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
