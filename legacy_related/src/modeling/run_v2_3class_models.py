"""Run leakage-safe Composite 3-class models on the fixed V2 splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs" / "leakage_safe_splits"
OUTPUT = ROOT / "outputs" / "v2_3class"
CLASSES = np.array([0, 1, 2])
CLASS_NAMES = {0: "low", 1: "middle", 2: "high"}

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
    numeric = frame.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in frame if column not in numeric]
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


def models() -> dict[str, object]:
    result: dict[str, object] = {
        "logistic": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=42,
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
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    return result


def evaluate(
    dataset: str,
    model: str,
    split: str,
    y_true: pd.Series,
    probability: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    prediction = CLASSES[np.argmax(probability, axis=1)]
    row: dict[str, object] = {
        "dataset": dataset,
        "model": model,
        "split": split,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision_macro": precision_score(
            y_true, prediction, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, prediction, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_true, prediction, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            y_true, prediction, average="weighted", zero_division=0
        ),
        "log_loss": log_loss(y_true, probability, labels=CLASSES),
    }
    for cls in CLASSES:
        key = CLASS_NAMES[int(cls)]
        row[f"precision_{key}"] = precision_score(
            y_true, prediction, labels=[cls], average="macro", zero_division=0
        )
        row[f"recall_{key}"] = recall_score(
            y_true, prediction, labels=[cls], average="macro", zero_division=0
        )
        row[f"f1_{key}"] = f1_score(
            y_true, prediction, labels=[cls], average="macro", zero_division=0
        )
    try:
        binary = label_binarize(y_true, classes=CLASSES)
        row["roc_auc_ovr_macro"] = roc_auc_score(
            binary, probability, average="macro", multi_class="ovr"
        )
    except ValueError:
        row["roc_auc_ovr_macro"] = np.nan

    matrix = confusion_matrix(y_true, prediction, labels=CLASSES)
    matrix_rows = []
    for true_class, values in zip(CLASSES, matrix):
        for predicted_class, count in zip(CLASSES, values):
            matrix_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "split": split,
                    "true_class": int(true_class),
                    "predicted_class": int(predicted_class),
                    "count": int(count),
                }
            )
    return row, matrix_rows


def run_dataset(
    name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[pd.DataFrame]]:
    df = pd.read_csv(INPUT / f"{name}_labeled_splits.csv")
    features = [column for column in FEATURES[name] if column in df.columns]
    missing = sorted(set(FEATURES[name]) - set(features))
    if missing:
        raise KeyError(f"{name}: missing features: {missing}")

    split_frames = {
        split: df[df["split"].eq(split)].copy()
        for split in ("train", "valid", "test")
    }
    player_sets = {
        split: set(frame["player_key"]) for split, frame in split_frames.items()
    }
    if any(
        player_sets[left] & player_sets[right]
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    ):
        raise AssertionError(f"{name}: player leakage detected")

    target = "success_3class"
    train = split_frames["train"]
    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, estimator in models().items():
        pipeline = Pipeline(
            [("preprocessor", preprocessor(train[features])), ("model", estimator)]
        )
        fit_kwargs = {}
        if model_name == "xgb":
            fit_kwargs["model__sample_weight"] = compute_sample_weight(
                class_weight="balanced", y=train[target]
            )
        pipeline.fit(train[features], train[target], **fit_kwargs)

        for split in ("valid", "test"):
            frame = split_frames[split]
            probability = pipeline.predict_proba(frame[features])
            row, matrices = evaluate(
                name, model_name, split, frame[target], probability
            )
            metric_rows.append(row)
            confusion_rows.extend(matrices)

            prediction = CLASSES[np.argmax(probability, axis=1)]
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "dataset": name,
                        "model": model_name,
                        "split": split,
                        "player_key": frame["player_key"].values,
                        "true_class": frame[target].values,
                        "predicted_class": prediction,
                        "prob_low": probability[:, 0],
                        "prob_middle": probability[:, 1],
                        "prob_high": probability[:, 2],
                    }
                )
            )
    return metric_rows, confusion_rows, prediction_frames


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    matrices: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    for name in ("hitter", "sp", "rp"):
        role_metrics, role_matrices, role_predictions = run_dataset(name)
        metrics.extend(role_metrics)
        matrices.extend(role_matrices)
        predictions.extend(role_predictions)

    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(
        OUTPUT / "v2_3class_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(matrices).to_csv(
        OUTPUT / "v2_3class_confusion_matrices.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        OUTPUT / "v2_3class_predictions.csv", index=False, encoding="utf-8-sig"
    )

    test = metric_df[metric_df["split"].eq("test")]
    best = (
        test.sort_values(
            ["dataset", "f1_macro", "balanced_accuracy"],
            ascending=[True, False, False],
        )
        .groupby("dataset", as_index=False)
        .head(1)
    )
    best.to_csv(
        OUTPUT / "v2_3class_best_models.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "target": "success_3class",
        "class_mapping": {"0": "low", "1": "middle", "2": "high"},
        "label_rule": "Role-specific train Composite tertiles; fixed cutoffs applied to validation and test.",
        "selection_metric": "Test table summary uses macro-F1; test was not used for tuning.",
        "random_seed": 42,
        "xgboost_available": XGBClassifier is not None,
    }
    with (OUTPUT / "v2_3class_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(
        test.sort_values(["dataset", "f1_macro"], ascending=[True, False]).to_string(
            index=False
        )
    )
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
