
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.modeling.run_temporal_final_models import (
    ROLE_CONFIG,
    temporal_split,
)

OUT = ROOT / "outputs" / "paper" / "weight_grid"
DATA = ROOT / "data" / "processed_v2"

metrics = pd.read_csv(OUT / "weight_grid_metrics.csv")
preds = pd.read_csv(OUT / "weight_grid_predictions.csv")

WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)
EXPECTED_TEST_N = {"hitter": 120, "sp": 65, "rp": 88}


def make_labels(df, cfg, weight):
    train_mask = df["split"].eq("train")
    standardized = {}

    for col, short in [
        (cfg["war"], "war"),
        (cfg["playtime"], "playtime"),
    ]:
        median = df.loc[train_mask, col].median()
        filled_train = df.loc[train_mask, col].fillna(median)

        mean = filled_train.mean()
        std = filled_train.std(ddof=0)

        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid training std: {col}")

        standardized[short] = (
            df[col].fillna(median) - mean
        ) / std

    score = (
        weight * standardized["war"]
        + (1 - weight) * standardized["playtime"]
    )

    cutoff = score.loc[train_mask].quantile(
        1 - cfg["top"]
    )

    return (score >= cutoff).astype(int)


def require_close(actual, expected, name, role, weight):
    if not np.isclose(actual, expected, atol=1e-8, rtol=0):
        raise AssertionError(
            f"{role}, w={weight:.1f}: "
            f"{name} mismatch "
            f"({actual} vs {expected})"
        )


def main():
    if len(metrics) != 33:
        raise AssertionError(
            f"Expected 33 result rows, found {len(metrics)}"
        )

    diagnostics = []

    for role, cfg in ROLE_CONFIG.items():
        df = pd.read_csv(
            DATA / f"{role}_cohort_v2.csv"
        )

        df["split"] = temporal_split(df)
        df = df[df["split"].notna()].copy()

        base_labels = make_labels(df, cfg, 0.5)
        base_test = base_labels.loc[
            df["split"].eq("test")
        ].to_numpy()

        for weight in WEIGHTS:
            weight = float(weight)
            labels = make_labels(df, cfg, weight)

            train = df[df["split"].eq("train")]
            valid = df[df["split"].eq("valid")]
            test = df[df["split"].eq("test")]

            train_y = labels.loc[train.index].to_numpy()
            valid_y = labels.loc[valid.index].to_numpy()
            test_y = labels.loc[test.index].to_numpy()

            m = metrics[
                metrics["role"].eq(role)
                & np.isclose(metrics["war_weight"], weight)
            ]

            p = preds[
                preds["role"].eq(role)
                & np.isclose(preds["war_weight"], weight)
            ].reset_index(drop=True)

            if len(m) != 1 or len(p) != len(test):
                raise AssertionError(
                    f"Missing or duplicate condition: {role}, {weight}"
                )

            row = m.iloc[0]

            if len(test) != EXPECTED_TEST_N[role]:
                raise AssertionError(
                    f"Unexpected Test size: {role}"
                )

            # Verify that the predictions refer to the
            # same player-contract rows in the same order.
            expected_ids = list(
                test[["player_key", "fa_year"]]
                .itertuples(index=False, name=None)
            )
            predicted_ids = list(
                p[["player_key", "fa_year"]]
                .itertuples(index=False, name=None)
            )

            if expected_ids != predicted_ids:
                raise AssertionError(
                    f"Test row alignment mismatch: {role}, {weight}"
                )

            y_true = p["y_true"].to_numpy(dtype=int)
            y_pred = p["y_pred"].to_numpy(dtype=int)
            prob = p["prob_high"].to_numpy(dtype=float)
            threshold = float(row["threshold"])

            if not np.array_equal(y_true, test_y):
                raise AssertionError(
                    f"Label mismatch: {role}, {weight}"
                )

            if not np.array_equal(
                y_pred,
                (prob >= threshold).astype(int),
            ):
                raise AssertionError(
                    f"Threshold mismatch: {role}, {weight}"
                )

            calculated = {
                "test_accuracy": accuracy_score(
                    y_true, y_pred
                ),
                "high_f1": f1_score(
                    y_true, y_pred, zero_division=0
                ),
                "precision": precision_score(
                    y_true, y_pred, zero_division=0
                ),
                "recall": recall_score(
                    y_true, y_pred, zero_division=0
                ),
                "pr_auc": average_precision_score(
                    y_true, prob
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y_true, y_pred
                ),
                "majority_accuracy": max(
                    np.mean(y_true == 0),
                    np.mean(y_true == 1),
                ),
                "label_agreement": np.mean(
                    y_true == base_test
                ),
            }

            for metric, value in calculated.items():
                require_close(
                    value,
                    float(row[metric]),
                    metric, role, weight,
                )

            valid_single_class = (
                len(np.unique(valid_y)) < 2
            )

            diagnostics.append({
                "role": role,
                "war_weight": weight,
                "train_n": len(train_y),
                "train_high": int(train_y.sum()),
                "valid_n": len(valid_y),
                "valid_high": int(valid_y.sum()),
                "valid_single_class": valid_single_class,
                "test_n": len(test_y),
                "test_high": int(test_y.sum()),
                "predicted_high": int(y_pred.sum()),
                "test_high_ratio": float(test_y.mean()),
                "threshold": threshold,
                "accuracy": calculated["test_accuracy"],
                "high_f1": calculated["high_f1"],
                "pr_auc": calculated["pr_auc"],
                "majority_accuracy": calculated[
                    "majority_accuracy"
                ],
                "accuracy_minus_majority": (
                    calculated["test_accuracy"]
                    - calculated["majority_accuracy"]
                ),
                "label_agreement": calculated[
                    "label_agreement"
                ],
            })

    report = pd.DataFrame(diagnostics)
    report.to_csv(
        OUT / "weight_grid_diagnostics.csv",
        index=False, encoding="utf-8-sig"
    )

    print("\n=== VERIFICATION ===")
    print("All 33 conditions: metric/label checks PASS")

    print("\n=== CLASS DISTRIBUTIONS ===")
    cols = [
        "role", "war_weight", "valid_n",
        "valid_high", "test_high",
        "predicted_high", "accuracy",
        "high_f1", "majority_accuracy",
    ]
    print(report[cols].round(3).to_string(index=False))

    print("\n=== SINGLE-CLASS VALIDATION CONDITIONS ===")
    bad_valid = report[report["valid_single_class"]]

    if bad_valid.empty:
        print("None")
    else:
        print(
            bad_valid[
                ["role", "war_weight",
                 "valid_n", "valid_high", "threshold"]
            ].to_string(index=False)
        )

    print("\n=== BELOW MAJORITY BASELINE ===")
    below = report[
        report["accuracy_minus_majority"] < -1e-9
    ]

    if below.empty:
        print("None")
    else:
        print(
            below[
                ["role", "war_weight", "accuracy",
                 "majority_accuracy", "high_f1"]
            ].round(3).to_string(index=False)
        )

    print("\nSaved:", OUT / "weight_grid_diagnostics.csv")


if __name__ == "__main__":
    main()
