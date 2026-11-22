
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.modeling.run_temporal_final_models import (
    ROLE_CONFIG,
    temporal_split,
    make_preprocessor,
    models,
    select_threshold,
)

DATA = ROOT / "data" / "processed_v2"
OUTPUT = ROOT / "outputs" / "paper" / "weight_grid"
OUTPUT.mkdir(parents=True, exist_ok=True)

WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)

PAPER_MODELS = json.loads(
    (ROOT / "config" / "paper_model_spec.json")
    .read_text(encoding="utf-8")
)

EXPECTED = {
    "hitter": (120, 0.783, 0.618),
    "sp": (65, 0.723, 0.308),
    "rp": (88, 0.898, 0.000),
}

ROLE_NAMES = {
    "hitter": "Hitter",
    "sp": "Starting Pitcher",
    "rp": "Relief Pitcher",
}


def make_labels(df, cfg, war_weight):
    result = df.copy()
    train_mask = result["split"].eq("train")

    for column, name in [
        (cfg["war"], "war"),
        (cfg["playtime"], "playtime"),
    ]:
        train_values = result.loc[train_mask, column]
        median = train_values.median()
        filled_train = train_values.fillna(median)

        mean = filled_train.mean()
        std = filled_train.std(ddof=0)

        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid std for {column}")

        result[f"z_{name}"] = (
            result[column].fillna(median) - mean
        ) / std

    result["composite_score"] = (
        war_weight * result["z_war"]
        + (1 - war_weight) * result["z_playtime"]
    )

    cutoff = result.loc[
        train_mask, "composite_score"
    ].quantile(1 - cfg["top"])

    result["y"] = (
        result["composite_score"] >= cutoff
    ).astype(int)

    return result


def evaluate_one(role, cfg, df, weight, baseline_labels):
    frame = make_labels(df, cfg, weight)

    train = frame[frame["split"] == "train"]
    valid = frame[frame["split"] == "valid"]
    test = frame[frame["split"] == "test"]

    features = cfg["features"]

    model_name = PAPER_MODELS[role]["2class"]["composite"]

    if model_name not in ("logistic", "rf"):
        raise ValueError(f"Unexpected paper model: {model_name}")

    estimator = clone(
        models("2class", train["y"])[model_name]
    )

    pipeline = Pipeline([
        ("preprocessor", make_preprocessor(train[features])),
        ("model", estimator),
    ])

    pipeline.fit(train[features], train["y"])

    valid_prob = pipeline.predict_proba(
        valid[features]
    )[:, 1]

    threshold = select_threshold(
        valid["y"], valid_prob
    )

    valid_pred = (valid_prob >= threshold).astype(int)

    test_prob = pipeline.predict_proba(
        test[features]
    )[:, 1]

    test_pred = (test_prob >= threshold).astype(int)

    y = test["y"].to_numpy()

    majority_accuracy = max(
        (y == 0).mean(), (y == 1).mean()
    )

    summary = {
        "role": role,
        "model": model_name,
        "war_weight": weight,
        "utilization_weight": round(1 - weight, 1),
        "n_test": len(test),
        "n_high_test": int(y.sum()),
        "threshold": threshold,
        "valid_accuracy": accuracy_score(
            valid["y"], valid_pred
        ),
        "test_accuracy": accuracy_score(y, test_pred),
        "high_f1": f1_score(
            y, test_pred, zero_division=0
        ),
        "precision": precision_score(
            y, test_pred, zero_division=0
        ),
        "recall": recall_score(
            y, test_pred, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y, test_prob
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y, test_pred
        ),
        "majority_accuracy": majority_accuracy,
        "label_agreement": (
            y == baseline_labels
        ).mean(),
    }

    predictions = test[
        ["player_key", "fa_year"]
    ].copy()

    predictions["role"] = role
    predictions["war_weight"] = weight
    predictions["y_true"] = y
    predictions["y_pred"] = test_pred
    predictions["prob_high"] = test_prob
    predictions["threshold"] = threshold

    return summary, predictions


def check_baseline(summary):
    baseline = summary[
        np.isclose(summary["war_weight"], 0.5)
    ]

    errors = []

    for role, (n, acc, f1) in EXPECTED.items():
        row = baseline[baseline["role"] == role]

        if len(row) != 1:
            errors.append(f"{role}: missing baseline")
            continue

        r = row.iloc[0]

        if (
            int(r["n_test"]) != n
            or round(float(r["test_accuracy"]), 3) != acc
            or round(float(r["high_f1"]), 3) != f1
        ):
            errors.append(
                f"{role}: n={r['n_test']}, "
                f"Acc={r['test_accuracy']:.3f}, "
                f"F1={r['high_f1']:.3f}"
            )

    if errors:
        raise RuntimeError(
            "Baseline reproduction failed:\n"
            + "\n".join(errors)
        )

    print("Baseline reproduction: PASS")


def make_plot(summary, metric, filename):
    plt.figure(figsize=(9, 5))

    for role, name in ROLE_NAMES.items():
        rows = summary[
            summary["role"] == role
        ].sort_values("war_weight")

        plt.plot(
            rows["war_weight"],
            rows[metric],
            marker="o",
            label=name,
        )

    plt.axvline(0.5, linestyle="--", linewidth=1)
    plt.xticks(WEIGHTS)
    plt.xlabel("WAR weight")
    plt.ylabel(metric)
    plt.title(f"{metric} across WAR weights")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=220)
    plt.close()


def main():
    summaries = []
    prediction_frames = []

    for role, cfg in ROLE_CONFIG.items():
        df = pd.read_csv(
            DATA / f"{role}_cohort_v2.csv"
        )

        df["split"] = temporal_split(df)
        df = df[df["split"].notna()].copy()

        baseline_frame = make_labels(df, cfg, 0.5)

        baseline_labels = baseline_frame.loc[
            baseline_frame["split"] == "test", "y"
        ].to_numpy()

        for weight in WEIGHTS:
            row, predictions = evaluate_one(
                role, cfg, df, float(weight),
                baseline_labels,
            )

            summaries.append(row)
            prediction_frames.append(predictions)

    summary = pd.DataFrame(summaries)

    # Compare the original 0.5:0.5 results first.
    check_baseline(summary)

    summary = summary.sort_values(
        ["role", "war_weight"]
    ).reset_index(drop=True)

    summary.to_csv(
        OUTPUT / "weight_grid_metrics.csv",
        index=False,
    )

    pd.concat(
        prediction_frames, ignore_index=True
    ).to_csv(
        OUTPUT / "weight_grid_predictions.csv",
        index=False,
    )

    make_plot(
        summary, "test_accuracy",
        "accuracy_vs_weight.png",
    )

    make_plot(
        summary, "high_f1",
        "high_f1_vs_weight.png",
    )

    print("\nAll 33 conditions:")
    print(
        summary[
            ["role", "war_weight", "test_accuracy",
             "high_f1", "pr_auc", "label_agreement"]
        ].round(3).to_string(index=False)
    )

    print("\nSaved to:", OUTPUT)


if __name__ == "__main__":
    main()
