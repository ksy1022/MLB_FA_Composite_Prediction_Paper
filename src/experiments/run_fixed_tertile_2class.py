# -*- coding: utf-8 -*-
"""
Fixed-tertile 2-class experiment for the MLB FA Composite paper.

Purpose
-------
Evaluate a SINGLE pre-specified binary outcome definition:

    High  = top tertile of the TRAINING distribution
    Other = lower two tertiles

The same rule is used for Hitter / Starting Pitcher / Relief Pitcher and for
both Composite and WAR-only outcomes.

This script does NOT search over outcome cutoffs.

Temporal design
---------------
Train      : 2012-2017
Validation : 2018-2019
Test       : 2020-2021

All label transformations/cut points are estimated from Train only.
For each role/outcome, LR/RF/XGB are trained on Train.
The probability threshold and representative model are selected using
Validation High-class F1. The locked model/threshold are then evaluated on Test.

Important:
The historical 2020-2021 test cohort has already been inspected in this project,
so it should not be described as a never-seen holdout. This script nevertheless
does not use Test performance to select the cutoff, model, or probability
threshold.

Outputs
-------
outputs/fixed_tertile_2class/
    validation_all_models.csv
    selected_model_config.json
    selected_valid_test_metrics.csv
    selected_test_metrics.csv
    class_counts.csv
    label_parameters.json
    selected_predictions.csv
    selected_confusion_matrices.csv
    FINAL_SUMMARY.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from src.modeling.run_temporal_final_models import (
    DATA,
    ROLE_CONFIG,
    binary_metrics,
    make_preprocessor,
    models,
    select_threshold,
    temporal_split,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "fixed_tertile_2class"
TOP_RATIO = 1.0 / 3.0

ROLE_LABELS = {
    "hitter": "Hitter",
    "sp": "Starting Pitcher",
    "rp": "Relief Pitcher",
}

# Deterministic tie-break after Validation F1.
MODEL_SIMPLICITY = {"logistic": 0, "rf": 1, "xgb": 2}


def fit_fixed_tertile_outcomes(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, dict]:
    """Build Composite and WAR labels from Train-only upper-tertile cutoffs."""
    out = df.copy()
    train_mask = out["split"].eq("train")
    params: dict[str, float] = {"top_ratio": TOP_RATIO}

    # Same Train-only standardization used by the current paper pipeline.
    for column, short in ((cfg["war"], "war"), (cfg["playtime"], "playtime")):
        median = float(out.loc[train_mask, column].median())
        filled_train = out.loc[train_mask, column].fillna(median)
        mean = float(filled_train.mean())
        std = float(filled_train.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid training SD for {column}: {std}")

        out[f"z_{short}"] = (out[column].fillna(median) - mean) / std
        params.update({
            f"{short}_median": median,
            f"{short}_mean": mean,
            f"{short}_std": std,
        })

    # Existing main code uses zWAR + zUtil. Multiplying by 0.5 would not alter
    # ranks or binary membership, but we keep the exact current convention.
    out["composite_score"] = out["z_war"] + out["z_playtime"]

    for outcome, score_col in (
        ("composite", "composite_score"),
        ("war", cfg["war"]),
    ):
        train_score = out.loc[train_mask, score_col]
        cut = float(train_score.quantile(2.0 / 3.0))
        out[f"{outcome}_2class"] = out[score_col].ge(cut).astype(int)
        params[f"{outcome}_2class_cut"] = cut

    return out, params


def choose_model(valid_rows: pd.DataFrame) -> pd.Series:
    """Select representative model using Validation only."""
    ranked = valid_rows.copy()
    ranked["model_simplicity"] = ranked["model"].map(MODEL_SIMPLICITY).fillna(99)
    ranked = ranked.sort_values(
        ["f1", "balanced_accuracy", "pr_auc", "recall", "model_simplicity"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    all_valid_rows = []
    selected_metric_rows = []
    selected_prediction_frames = []
    confusion_rows = []
    count_rows = []
    label_params = {}
    selected_config = {}

    for role, cfg in ROLE_CONFIG.items():
        role_label = ROLE_LABELS[role]
        print("\n" + "=" * 78)
        print(role_label)
        print("=" * 78)

        df = pd.read_csv(DATA / f"{role}_cohort_v2.csv")
        df["fa_year"] = pd.to_numeric(df["fa_year"], errors="coerce")
        df["split"] = temporal_split(df)
        df = df[df["split"].notna()].copy()

        df, params = fit_fixed_tertile_outcomes(df, cfg)
        label_params[role] = params

        features = list(cfg["features"])
        train = df[df["split"].eq("train")].copy()
        valid = df[df["split"].eq("valid")].copy()
        test = df[df["split"].eq("test")].copy()

        for split_name, frame in (("train", train), ("valid", valid), ("test", test)):
            for outcome in ("composite", "war"):
                target = f"{outcome}_2class"
                count_rows.append({
                    "role": role,
                    "role_label": role_label,
                    "outcome": outcome,
                    "split": split_name,
                    "n": len(frame),
                    "n_players": (
                        frame["player_key"].nunique()
                        if "player_key" in frame.columns else len(frame)
                    ),
                    "high_n": int(frame[target].sum()),
                    "other_n": int((frame[target] == 0).sum()),
                    "high_rate": float(frame[target].mean()),
                })

        selected_config[role] = {}

        for outcome in ("composite", "war"):
            target = f"{outcome}_2class"
            fitted = {}
            valid_records = []

            for model_name, estimator in models("2class", train[target]).items():
                pipeline = Pipeline([
                    ("preprocessor", make_preprocessor(train[features])),
                    ("model", clone(estimator)),
                ])
                pipeline.fit(train[features], train[target])

                valid_prob = pipeline.predict_proba(valid[features])[:, 1]
                threshold = select_threshold(valid[target], valid_prob)
                values, _ = binary_metrics(valid[target], valid_prob, threshold)

                record = {
                    "role": role,
                    "role_label": role_label,
                    "outcome": outcome,
                    "model": model_name,
                    "split": "valid",
                    "n": len(valid),
                    **values,
                }
                valid_records.append(record)
                all_valid_rows.append(record)
                fitted[model_name] = (pipeline, threshold)

            valid_df = pd.DataFrame(valid_records)
            winner = choose_model(valid_df)
            model_name = str(winner["model"])
            pipeline, threshold = fitted[model_name]

            selected_config[role][outcome] = {
                "top_ratio": TOP_RATIO,
                "outcome_cut": params[f"{outcome}_2class_cut"],
                "model": model_name,
                "probability_threshold": float(threshold),
                "selection_metric": "validation_high_class_f1",
            }

            print(
                f"{outcome:>9s}: cut={params[f'{outcome}_2class_cut']:.4f} | "
                f"model={model_name} | threshold={threshold:.2f} | "
                f"valid F1={winner['f1']:.3f}"
            )

            for split_name, frame in (("valid", valid), ("test", test)):
                prob = pipeline.predict_proba(frame[features])[:, 1]
                values, pred = binary_metrics(frame[target], prob, threshold)

                selected_metric_rows.append({
                    "role": role,
                    "role_label": role_label,
                    "outcome": outcome,
                    "model": model_name,
                    "split": split_name,
                    "n": len(frame),
                    **values,
                })

                matrix = confusion_matrix(frame[target], pred, labels=[0, 1])
                for true_cls in (0, 1):
                    for pred_cls in (0, 1):
                        confusion_rows.append({
                            "role": role,
                            "role_label": role_label,
                            "outcome": outcome,
                            "model": model_name,
                            "split": split_name,
                            "true_class": true_cls,
                            "predicted_class": pred_cls,
                            "count": int(matrix[true_cls, pred_cls]),
                        })

                id_cols = [c for c in ["player_key", "player", "fa_year"] if c in frame.columns]
                pred_frame = frame[id_cols].reset_index(drop=True).copy()
                pred_frame["role"] = role
                pred_frame["role_label"] = role_label
                pred_frame["outcome"] = outcome
                pred_frame["model"] = model_name
                pred_frame["split"] = split_name
                pred_frame["threshold"] = threshold
                pred_frame["y_true"] = frame[target].to_numpy()
                pred_frame["y_pred"] = pred
                pred_frame["probability_high"] = prob
                selected_prediction_frames.append(pred_frame)

    valid_all = pd.DataFrame(all_valid_rows)
    selected_metrics = pd.DataFrame(selected_metric_rows)
    test_metrics = selected_metrics[selected_metrics["split"].eq("test")].copy()
    counts = pd.DataFrame(count_rows)
    confusions = pd.DataFrame(confusion_rows)
    predictions = pd.concat(selected_prediction_frames, ignore_index=True)

    valid_all.to_csv(OUT / "validation_all_models.csv", index=False, encoding="utf-8-sig")
    selected_metrics.to_csv(
        OUT / "selected_valid_test_metrics.csv", index=False, encoding="utf-8-sig"
    )
    test_metrics.to_csv(
        OUT / "selected_test_metrics.csv", index=False, encoding="utf-8-sig"
    )
    counts.to_csv(OUT / "class_counts.csv", index=False, encoding="utf-8-sig")
    confusions.to_csv(
        OUT / "selected_confusion_matrices.csv", index=False, encoding="utf-8-sig"
    )
    predictions.to_csv(
        OUT / "selected_predictions.csv", index=False, encoding="utf-8-sig"
    )

    with (OUT / "label_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(label_params, f, ensure_ascii=False, indent=2)

    with (OUT / "selected_model_config.json").open("w", encoding="utf-8") as f:
        json.dump(selected_config, f, ensure_ascii=False, indent=2)

    show_cols = [
        "role_label", "outcome", "model", "threshold",
        "accuracy", "balanced_accuracy", "precision", "recall", "f1", "pr_auc",
    ]
    summary = test_metrics[show_cols].sort_values(["role_label", "outcome"])

    class_show = counts[
        (counts["split"].eq("test")) & (counts["outcome"].eq("composite"))
    ][["role_label", "n", "high_n", "other_n", "high_rate"]]

    text = []
    text.append("FIXED TOP-TERTILE 2-CLASS EXPERIMENT")
    text.append("=" * 78)
    text.append("High = top 1/3 of TRAIN distribution; Other = lower 2/3")
    text.append("Model + probability threshold selected by Validation F1 only.")
    text.append("")
    text.append("TEST CLASS COUNTS (Composite)")
    text.append(class_show.to_string(index=False))
    text.append("")
    text.append("SELECTED TEST METRICS")
    text.append(summary.to_string(index=False))
    text.append("")
    text.append(
        "Caution: 2020-2021 has been inspected previously in this research project; "
        "do not describe it as a never-seen holdout."
    )

    final_text = "\n".join(text)
    (OUT / "FINAL_SUMMARY.txt").write_text(final_text, encoding="utf-8")

    print("\n" + final_text)
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
