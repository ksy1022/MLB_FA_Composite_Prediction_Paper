"""2-class Composite weight sensitivity: fixed paper models, out-of-time test.

Run from repository root:
    python src/paper/weight_performance_sensitivity_2class.py

The script imports the exact transformations, preprocessing, LR/RF hyperparameters,
and validation threshold selection from run_temporal_final_models.py.
Each weight produces its own train-fitted Composite labels and is retrained.
No test data is used to select a model, threshold, or weight.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             average_precision_score)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "modeling"))
import run_temporal_final_models as base  # noqa: E402

DATA = ROOT / "data" / "processed_v2"
SPEC = ROOT / "config" / "paper_model_spec.json"
OUT = ROOT / "outputs" / "paper"
WEIGHTS = [(0.4, 0.6), (0.5, 0.5), (0.6, 0.4)]
ROLES = ["hitter", "sp", "rp"]


def run_one(role: str, war_weight: float, util_weight: float,
            paper_spec: dict) -> tuple[dict, pd.DataFrame]:
    cfg = base.ROLE_CONFIG[role]
    df = pd.read_csv(DATA / f"{role}_cohort_v2.csv")
    df["split"] = base.temporal_split(df)
    df = df.loc[df["split"].notna()].copy()
    # Train-only median imputation and z-scoring, exactly as main study.
    df, _ = base.fit_outcomes(df, cfg)
    score = war_weight * df["z_war"] + util_weight * df["z_playtime"]

    # Recalculate Train cutoff for each alternative weight.
    train_mask = df["split"].eq("train")
    cutoff = float(score.loc[train_mask].quantile(1.0 - cfg["top"]))
    df["weighted_target"] = score.ge(cutoff).astype(int)

    train = df.loc[df["split"].eq("train")]
    valid = df.loc[df["split"].eq("valid")]
    test = df.loc[df["split"].eq("test")]
    features = cfg["features"]
    model_name = paper_spec[role]["2class"]["composite"]
    estimator = base.models("2class", train["weighted_target"])[model_name]
    pipe = Pipeline([
        ("preprocessor", base.make_preprocessor(train[features])),
        ("model", clone(estimator)),
    ])
    pipe.fit(train[features], train["weighted_target"])

    valid_prob = pipe.predict_proba(valid[features])[:, 1]
    # Replicate original Validation Accuracy, Balanced Accuracy tie-break.
    threshold = base.select_threshold(valid["weighted_target"], valid_prob)
    test_prob = pipe.predict_proba(test[features])[:, 1]
    y_true = test["weighted_target"].to_numpy().astype(int)
    y_pred = (test_prob >= threshold).astype(int)
    majority_baseline = max(np.mean(y_true == 0), np.mean(y_true == 1))

    metrics = {
        "role": role,
        "model": model_name,
        "war_weight": war_weight,
        "util_weight": util_weight,
        "train_n": len(train), "valid_n": len(valid), "test_n": len(test),
        "train_high_n": int(train["weighted_target"].sum()),
        "valid_high_n": int(valid["weighted_target"].sum()),
        "test_high_n": int(y_true.sum()),
        "train_cutoff": cutoff,
        "threshold": float(threshold),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_high_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_high_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "test_high_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_pr_auc": float(average_precision_score(y_true, test_prob)),
        "test_majority_baseline_accuracy": float(majority_baseline),
        "test_predicted_high_n": int(y_pred.sum()),
    }
    prediction = test[["player_key", "fa_year"]].copy()
    prediction["role"] = role
    prediction["war_weight"] = war_weight
    prediction["util_weight"] = util_weight
    prediction["y_true"] = y_true
    prediction["y_pred"] = y_pred
    prediction["prob_high"] = test_prob
    return metrics, prediction


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paper_spec = json.loads(SPEC.read_text(encoding="utf-8"))
    metrics, predictions = [], []
    for role in ROLES:
        for war_w, util_w in WEIGHTS:
            m, p = run_one(role, war_w, util_w, paper_spec)
            metrics.append(m)
            predictions.append(p)
    result = pd.DataFrame(metrics)
    preds = pd.concat(predictions, ignore_index=True)

    for role in ROLES:
        role_rows = result["role"].eq(role)
        base_row = result.loc[role_rows & result["war_weight"].eq(.5)].iloc[0]
        result.loc[role_rows, "accuracy_delta_vs_equal"] = (
            result.loc[role_rows, "test_accuracy"] - base_row["test_accuracy"]
        )
        result.loc[role_rows, "high_f1_delta_vs_equal"] = (
            result.loc[role_rows, "test_high_f1"] - base_row["test_high_f1"]
        )
        base_label = preds.loc[
            preds["role"].eq(role) & preds["war_weight"].eq(.5),
            ["player_key", "fa_year", "y_true"],
        ].rename(columns={"y_true": "base_y_true"})
        for war_w, _ in WEIGHTS:
            mask = preds["role"].eq(role) & preds["war_weight"].eq(war_w)
            candidate = preds.loc[mask, ["player_key", "fa_year", "y_true"]]
            merged = candidate.merge(base_label, on=["player_key", "fa_year"], validate="one_to_one")
            result.loc[role_rows & result["war_weight"].eq(war_w), "test_label_agreement_vs_equal"] = float(
                (merged["y_true"] == merged["base_y_true"]).mean()
            )

    # If rerun from the reference repository, ensure exact baseline replication.
    reference_path = OUT / "table5_binary_composite.csv"
    if reference_path.exists():
        ref = pd.read_csv(reference_path)
        mapping = {"hitter": "Hitter", "sp": "Starting Pitcher", "rp": "Relief Pitcher"}
        for role in ROLES:
            original = ref.loc[ref["Role"].eq(mapping[role])].iloc[0]
            current = result.loc[result["role"].eq(role) & result["war_weight"].eq(.5)].iloc[0]
            if not (np.isclose(current["test_accuracy"], original["Accuracy"], atol=1e-9)
                    and np.isclose(current["test_high_f1"], original["F1"], atol=1e-9)
                    and np.isclose(current["threshold"], original["Threshold"], atol=1e-9)):
                raise AssertionError(f"Baseline failed to reproduce original paper result: {role}")

    result.to_csv(OUT / "weight_performance_sensitivity_2class.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT / "weight_performance_predictions_2class.csv", index=False, encoding="utf-8-sig")
    print("Baseline reproduction: PASS")
    cols = ["role", "war_weight", "util_weight", "model", "threshold", "test_n",
            "test_high_n", "test_accuracy", "test_high_f1",
            "accuracy_delta_vs_equal", "high_f1_delta_vs_equal",
            "test_label_agreement_vs_equal"]
    print(result[cols].round(4).to_string(index=False))
    print("Saved to", OUT)


if __name__ == "__main__":
    main()
