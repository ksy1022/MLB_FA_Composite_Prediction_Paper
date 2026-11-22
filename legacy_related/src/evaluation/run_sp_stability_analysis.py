"""SP 2-class stability analysis with player-level repeated splits.

Run from the project root:
    python src/evaluation/run_sp_stability_analysis.py

The script searches the project-standard input location first, but an explicit
CSV can be supplied with --input.
"""
from __future__ import annotations

import argparse
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
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

RANDOM_SEEDS = list(range(30))
TARGET = "success_2class"
ID_COLS = {"player_key", "player", "player_norm", "split", "role"}
LEAKAGE_PREFIXES = ("post_", "z_post_", "composite_", "success_")


def locate_input(project_root: Path, explicit: str | None) -> Path:
    candidates = [
        Path(explicit) if explicit else None,
        project_root / "outputs/leakage_safe_splits/sp_labeled_splits.csv",
        project_root / "upload/sp_labeled_splits.csv",
    ]
    for path in candidates:
        if path and path.exists():
            return path
    raise FileNotFoundError("sp_labeled_splits.csv를 찾지 못했습니다. --input으로 지정하세요.")


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    allowed = [
        c for c in df.columns
        if c not in ID_COLS
        and c != TARGET
        and not c.startswith(LEAKAGE_PREFIXES)
        and c not in {"is_real_fa", "anchor_year"}
    ]
    categorical = [c for c in allowed if df[c].dtype == "object"]
    numeric = [c for c in allowed if c not in categorical]
    return numeric, categorical


def preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def models(seed: int) -> dict[str, object]:
    result = {
        "logistic": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        result["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=seed, n_jobs=-1,
        )
    return result


def best_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = f1_score(y, prob >= threshold, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(threshold), float(score)
    return best_t, best_f1


def metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output", default="outputs/v2_sp_stability")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    input_path = locate_input(root, args.input)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    numeric, categorical = feature_columns(df)
    features = numeric + categorical

    development = df[df["split"].isin(["train", "valid"])].copy()
    fixed_test = df[df["split"] == "test"].copy()
    rows = []

    for seed in RANDOM_SEEDS:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_idx, valid_idx = next(splitter.split(
            development, development[TARGET], groups=development["player_key"]
        ))
        train, valid = development.iloc[train_idx], development.iloc[valid_idx]
        for model_name, estimator in models(seed).items():
            pipe = Pipeline([
                ("prep", preprocessor(numeric, categorical)),
                ("model", estimator),
            ])
            pipe.fit(train[features], train[TARGET])
            valid_prob = pipe.predict_proba(valid[features])[:, 1]
            threshold, _ = best_threshold(valid[TARGET].to_numpy(), valid_prob)
            row = {
                "seed": seed,
                "model": model_name,
                "train_n": len(train),
                "valid_n": len(valid),
                "valid_positive_n": int(valid[TARGET].sum()),
                "valid_positive_rate": float(valid[TARGET].mean()),
            }
            row.update({f"valid_{k}": v for k, v in metrics(
                valid[TARGET].to_numpy(), valid_prob, threshold
            ).items()})
            rows.append(row)

    repeated = pd.DataFrame(rows)
    repeated.to_csv(output / "sp_repeated_group_split_metrics.csv", index=False)
    summary = repeated.groupby("model").agg(
        n_runs=("seed", "count"),
        positive_n_min=("valid_positive_n", "min"),
        positive_n_median=("valid_positive_n", "median"),
        positive_n_max=("valid_positive_n", "max"),
        threshold_mean=("valid_threshold", "mean"),
        threshold_sd=("valid_threshold", "std"),
        f1_mean=("valid_f1", "mean"),
        f1_sd=("valid_f1", "std"),
        f1_min=("valid_f1", "min"),
        f1_max=("valid_f1", "max"),
        pr_auc_mean=("valid_pr_auc", "mean"),
        pr_auc_sd=("valid_pr_auc", "std"),
        recall_mean=("valid_recall", "mean"),
        recall_sd=("valid_recall", "std"),
    ).reset_index()
    summary.to_csv(output / "sp_stability_summary.csv", index=False)

    # Choose a robust threshold as the median of repeated validation optima,
    # refit on all development rows, then evaluate the untouched fixed test once.
    test_rows = []
    for model_name, estimator in models(2026).items():
        threshold = float(repeated.loc[repeated["model"] == model_name, "valid_threshold"].median())
        pipe = Pipeline([
            ("prep", preprocessor(numeric, categorical)),
            ("model", estimator),
        ])
        pipe.fit(development[features], development[TARGET])
        prob = pipe.predict_proba(fixed_test[features])[:, 1]
        row = {
            "model": model_name,
            "development_n": len(development),
            "test_n": len(fixed_test),
            "test_positive_n": int(fixed_test[TARGET].sum()),
            "test_positive_rate": float(fixed_test[TARGET].mean()),
        }
        row.update(metrics(fixed_test[TARGET].to_numpy(), prob, threshold))
        test_rows.append(row)
    pd.DataFrame(test_rows).to_csv(output / "sp_fixed_test_robust_threshold.csv", index=False)

    metadata = {
        "input": str(input_path),
        "features": features,
        "excluded_by_design": sorted(ID_COLS | {TARGET, "is_real_fa", "anchor_year"}),
        "seeds": RANDOM_SEEDS,
        "xgboost_available": XGBClassifier is not None,
        "group_key": "player_key",
        "fixed_test_untouched_until_final_step": True,
    }
    (output / "sp_stability_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
