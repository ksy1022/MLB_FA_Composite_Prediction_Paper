"""Run the final out-of-time classification experiment on the V2 cohorts.

Temporal split
--------------
train: 2012--2017
validation: 2018--2019
test: 2020--2021

All outcome transformations and cut points are estimated on the training
partition only. Binary probability thresholds and final models are selected
on validation accuracy and then applied once to the test partition.
"""

from __future__ import annotations

import json
from pathlib import Path

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
DATA = ROOT / "data" / "processed_v2"
OUTPUT = ROOT / "outputs" / "temporal_accuracy_final"
CLASSES = np.array([0, 1, 2])

ROLE_CONFIG = {
    "hitter": {
        "war": "post_WAR_3y_sum",
        "playtime": "post_PA_3y",
        "top": 0.40,
        "features": [
            "Pos_group", "age_at_contract", "pre_PA_3y", "pre_WAR_3y_avg",
            "pre_WAR_3y_std", "pre_OPSplus_3y_avg", "pre_BB_rate_3y_avg",
            "pre_K_rate_3y_avg", "AAV", "Yrs",
        ],
    },
    "sp": {
        "war": "post_WAR_3y_sum",
        "playtime": "post_IP_3y",
        "top": 0.20,
        "features": [
            "age_at_contract", "pre_IP_3y", "pre_WAR_3y_avg",
            "pre_WAR_3y_std", "pre_ERA_3y_avg", "pre_FIP_3y_avg",
            "pre_KBB_3y_avg", "pre_GS_3y_sum", "pre_G_3y_sum", "AAV", "Yrs",
        ],
    },
    "rp": {
        "war": "post_WAR_3y_sum",
        "playtime": "post_IP_3y",
        "top": 0.20,
        "features": [
            "age_at_contract", "pre_IP_3y", "pre_WAR_3y_avg",
            "pre_WAR_3y_std", "pre_ERA_3y_avg", "pre_FIP_3y_avg",
            "pre_KBB_3y_avg", "pre_GS_3y_sum", "pre_G_3y_sum", "AAV", "Yrs",
        ],
    },
}


def temporal_split(df: pd.DataFrame) -> pd.Series:
    split = pd.Series(index=df.index, dtype="object")
    split[df["fa_year"].le(2017)] = "train"
    split[df["fa_year"].between(2018, 2019)] = "valid"
    split[df["fa_year"].between(2020, 2021)] = "test"
    return split


def fit_outcomes(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    train = out["split"].eq("train")
    params: dict[str, float] = {}

    for column, short in ((cfg["war"], "war"), (cfg["playtime"], "playtime")):
        median = float(out.loc[train, column].median())
        filled_train = out.loc[train, column].fillna(median)
        mean = float(filled_train.mean())
        std = float(filled_train.std(ddof=0))
        if std == 0:
            raise ValueError(f"Training standard deviation is zero: {column}")
        out[f"z_{short}"] = (out[column].fillna(median) - mean) / std
        params.update({
            f"{short}_median": median,
            f"{short}_mean": mean,
            f"{short}_std": std,
        })

    out["composite_score"] = (
    0.5 * out["z_war"]
    + 0.5 * out["z_playtime"]
    )
    for outcome, score in (("composite", "composite_score"), ("war", cfg["war"])):
        train_score = out.loc[train, score]
        cut2 = float(train_score.quantile(1 - cfg["top"]))
        low = float(train_score.quantile(1 / 3))
        high = float(train_score.quantile(2 / 3))
        out[f"{outcome}_2class"] = out[score].ge(cut2).astype(int)
        out[f"{outcome}_3class"] = np.select(
            [out[score].lt(low), out[score].lt(high)], [0, 1], default=2
        ).astype(int)
        params.update({
            f"{outcome}_2class_cut": cut2,
            f"{outcome}_3class_low_cut": low,
            f"{outcome}_3class_high_cut": high,
        })
    return out, params


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def models(task: str, y: pd.Series) -> dict[str, object]:
    result: dict[str, object] = {
        "logistic": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=4000, random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=1,
        ),
    }
    if XGBClassifier is not None:
        common = dict(
            n_estimators=300, learning_rate=0.03, max_depth=3,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, n_jobs=1,
        )
        if task == "2class":
            positives = int(y.sum())
            negatives = len(y) - positives
            result["xgb"] = XGBClassifier(
                **common,
                scale_pos_weight=(negatives / positives if positives else 1.0),
                eval_metric="logloss",
            )
        else:
            result["xgb"] = XGBClassifier(
                **common, objective="multi:softprob", num_class=3,
                eval_metric="mlogloss",
            )
    return result


def select_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    candidates = np.arange(0.05, 0.951, 0.01)
    rows = []
    for threshold in candidates:
        prediction = probability >= threshold
        rows.append((
            accuracy_score(y_true, prediction),
            balanced_accuracy_score(y_true, prediction),
            f1_score(y_true, prediction, zero_division=0),
            -abs(threshold - 0.5),
            threshold,
        ))
    return float(max(rows)[-1])


def binary_metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> tuple[dict, np.ndarray]:
    prediction = (probability >= threshold).astype(int)
    row = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "log_loss": log_loss(y_true, probability, labels=[0, 1]),
    }
    return row, prediction


def multiclass_metrics(y_true: pd.Series, probability: np.ndarray) -> tuple[dict, np.ndarray]:
    prediction = np.argmax(probability, axis=1)
    row = {
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
    binary = label_binarize(y_true, classes=CLASSES)
    row["roc_auc_ovr_macro"] = roc_auc_score(
        binary, probability, average="macro", multi_class="ovr"
    )
    for cls, name in enumerate(("low", "middle", "high")):
        row[f"precision_{name}"] = precision_score(
            y_true, prediction, labels=[cls], average="macro", zero_division=0
        )
        row[f"recall_{name}"] = recall_score(
            y_true, prediction, labels=[cls], average="macro", zero_division=0
        )
        row[f"f1_{name}"] = f1_score(
            y_true, prediction, labels=[cls], average="macro", zero_division=0
        )
    return row, prediction


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_frames, confusion_rows, count_rows = [], [], [], []
    baseline_rows = []
    label_parameters = {}

    for role, cfg in ROLE_CONFIG.items():
        df = pd.read_csv(DATA / f"{role}_cohort_v2.csv")
        df["split"] = temporal_split(df)
        df = df[df["split"].notna()].copy()
        df, params = fit_outcomes(df, cfg)
        label_parameters[role] = params
        features = cfg["features"]

        for split_name, frame in df.groupby("split"):
            for outcome in ("composite", "war"):
                binary_target = f"{outcome}_2class"
                multiclass_target = f"{outcome}_3class"
                binary_majority = int(frame[binary_target].mode().iloc[0])
                multiclass_majority = int(frame[multiclass_target].mode().iloc[0])
                baseline_rows.extend([
                    {
                        "role": role, "outcome": outcome, "task": "2class",
                        "split": split_name, "n": len(frame),
                        "majority_class": binary_majority,
                        "majority_accuracy": accuracy_score(
                            frame[binary_target],
                            np.full(len(frame), binary_majority),
                        ),
                    },
                    {
                        "role": role, "outcome": outcome, "task": "3class",
                        "split": split_name, "n": len(frame),
                        "majority_class": multiclass_majority,
                        "majority_accuracy": accuracy_score(
                            frame[multiclass_target],
                            np.full(len(frame), multiclass_majority),
                        ),
                    },
                ])
                count_rows.append({
                    "role": role,
                    "split": split_name,
                    "n": len(frame),
                    "n_players": frame["player_key"].nunique(),
                    "min_year": int(frame["fa_year"].min()),
                    "max_year": int(frame["fa_year"].max()),
                    "outcome": outcome,
                    "positive_2class": int(frame[f"{outcome}_2class"].sum()),
                    "negative_2class": int((frame[f"{outcome}_2class"] == 0).sum()),
                    "low_3class": int((frame[f"{outcome}_3class"] == 0).sum()),
                    "middle_3class": int((frame[f"{outcome}_3class"] == 1).sum()),
                    "high_3class": int((frame[f"{outcome}_3class"] == 2).sum()),
                })

        train = df[df["split"].eq("train")]
        valid = df[df["split"].eq("valid")]
        test = df[df["split"].eq("test")]

        for outcome in ("composite", "war"):
            for task in ("2class", "3class"):
                target = f"{outcome}_{task}"
                for model_name, estimator in models(task, train[target]).items():
                    pipeline = Pipeline([
                        ("preprocessor", make_preprocessor(train[features])),
                        ("model", clone(estimator)),
                    ])
                    fit_kwargs = {}
                    if model_name == "xgb" and task == "3class":
                        fit_kwargs["model__sample_weight"] = compute_sample_weight(
                            class_weight="balanced", y=train[target]
                        )
                    pipeline.fit(train[features], train[target], **fit_kwargs)
                    valid_probability = pipeline.predict_proba(valid[features])
                    test_probability = pipeline.predict_proba(test[features])
                    threshold = (
                        select_threshold(valid[target], valid_probability[:, 1])
                        if task == "2class" else None
                    )

                    for split_name, frame, probability in (
                        ("valid", valid, valid_probability),
                        ("test", test, test_probability),
                    ):
                        if task == "2class":
                            values, prediction = binary_metrics(
                                frame[target], probability[:, 1], threshold
                            )
                        else:
                            values, prediction = multiclass_metrics(
                                frame[target], probability
                            )
                        metric_rows.append({
                            "role": role, "outcome": outcome, "task": task,
                            "model": model_name, "split": split_name,
                            "n": len(frame), **values,
                        })

                        matrix = confusion_matrix(
                            frame[target], prediction,
                            labels=[0, 1] if task == "2class" else CLASSES,
                        )
                        for true_class, values_by_prediction in enumerate(matrix):
                            for predicted_class, count in enumerate(values_by_prediction):
                                confusion_rows.append({
                                    "role": role, "outcome": outcome, "task": task,
                                    "model": model_name, "split": split_name,
                                    "true_class": true_class,
                                    "predicted_class": predicted_class,
                                    "count": int(count),
                                })

                        ids = frame[["player_key", "fa_year"]].reset_index(drop=True)
                        pred_frame = ids.assign(
                            role=role, outcome=outcome, task=task,
                            model=model_name, split=split_name,
                            y_true=frame[target].to_numpy(),
                            y_pred=prediction,
                        )
                        for column_index in range(probability.shape[1]):
                            pred_frame[f"probability_{column_index}"] = probability[:, column_index]
                        prediction_frames.append(pred_frame)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "temporal_metrics.csv", index=False)
    pd.DataFrame(count_rows).to_csv(OUTPUT / "temporal_class_counts.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(
        OUTPUT / "temporal_majority_baselines.csv", index=False
    )
    pd.DataFrame(confusion_rows).to_csv(
        OUTPUT / "temporal_confusion_matrices.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        OUTPUT / "temporal_predictions.csv", index=False
    )
    with (OUTPUT / "temporal_label_parameters.json").open("w", encoding="utf-8") as file:
        json.dump(label_parameters, file, ensure_ascii=False, indent=2)

    valid_metrics = metrics[metrics["split"].eq("valid")].copy()
    selected = (
        valid_metrics.sort_values(
            ["role", "outcome", "task", "accuracy", "balanced_accuracy"],
            ascending=[True, True, True, False, False],
        )
        .groupby(["role", "outcome", "task"], as_index=False)
        .head(1)[["role", "outcome", "task", "model"]]
    )
    primary = selected.merge(
        metrics[metrics["split"].eq("test")],
        on=["role", "outcome", "task", "model"],
        how="left",
    )
    test_baselines = pd.DataFrame(baseline_rows)
    test_baselines = test_baselines[test_baselines["split"].eq("test")][
        ["role", "outcome", "task", "majority_class", "majority_accuracy"]
    ]
    primary = primary.merge(
        test_baselines, on=["role", "outcome", "task"], how="left"
    )
    primary["accuracy_gain_over_majority"] = (
        primary["accuracy"] - primary["majority_accuracy"]
    )
    primary = primary.sort_values(["task", "role", "outcome"]).reset_index(drop=True)
    primary.to_csv(OUTPUT / "temporal_best_models.csv", index=False)


if __name__ == "__main__":
    run()
