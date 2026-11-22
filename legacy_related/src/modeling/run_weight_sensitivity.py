from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "temporal_run" / "run_temporal_final_models.py"
DATA_DIR = (
    HERE / "project_extract" / "Predicting_the_Performance_of_MLB_FA"
    / "data" / "processed_v2"
)
OUT_DIR = HERE / "weight_sensitivity_results"
WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]

spec = importlib.util.spec_from_file_location("temporal_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def weighted_labels(df: pd.DataFrame, cfg: dict, war_weight: float) -> pd.DataFrame:
    out = df.copy()
    train = out["split"].eq("train")
    for column, short in ((cfg["war"], "war"), (cfg["playtime"], "playtime")):
        median = float(out.loc[train, column].median())
        filled = out.loc[train, column].fillna(median)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        out[f"z_{short}"] = (out[column].fillna(median) - mean) / std

    out["weighted_composite"] = (
        war_weight * out["z_war"] + (1 - war_weight) * out["z_playtime"]
    )
    train_score = out.loc[train, "weighted_composite"]
    cut2 = float(train_score.quantile(1 - cfg["top"]))
    low = float(train_score.quantile(1 / 3))
    high = float(train_score.quantile(2 / 3))
    out["label_2class"] = out["weighted_composite"].ge(cut2).astype(int)
    out["label_3class"] = np.select(
        [out["weighted_composite"].lt(low), out["weighted_composite"].lt(high)],
        [0, 1],
        default=2,
    ).astype(int)
    return out


def fit_and_select(df: pd.DataFrame, cfg: dict, task: str) -> dict:
    target = f"label_{task}"
    features = cfg["features"]
    train = df["split"].eq("train")
    valid = df["split"].eq("valid")
    test = df["split"].eq("test")

    X_train, y_train = df.loc[train, features], df.loc[train, target]
    X_valid, y_valid = df.loc[valid, features], df.loc[valid, target]
    X_test, y_test = df.loc[test, features], df.loc[test, target]
    candidates = []

    for name, estimator in base.models(task, y_train).items():
        pipe = Pipeline([
            ("preprocessor", base.make_preprocessor(X_train)),
            ("model", clone(estimator)),
        ])
        fit_kwargs = {}
        if name == "xgb" and task == "3class":
            fit_kwargs["model__sample_weight"] = base.compute_sample_weight(
                class_weight="balanced", y=y_train
            )
        pipe.fit(X_train, y_train, **fit_kwargs)
        valid_prob = pipe.predict_proba(X_valid)

        if task == "2class":
            threshold = base.select_threshold(y_valid, valid_prob[:, 1])
            valid_pred = (valid_prob[:, 1] >= threshold).astype(int)
        else:
            threshold = np.nan
            valid_pred = np.argmax(valid_prob, axis=1)

        candidates.append({
            "model": name,
            "pipeline": pipe,
            "threshold": threshold,
            "valid_accuracy": accuracy_score(y_valid, valid_pred),
        })

    selected = max(candidates, key=lambda row: row["valid_accuracy"])
    probability = selected["pipeline"].predict_proba(X_test)
    if task == "2class":
        metrics, _ = base.binary_metrics(
            y_test, probability[:, 1], selected["threshold"]
        )
        performance = metrics["f1"]
        metric_name = "High-class F1"
    else:
        metrics, _ = base.multiclass_metrics(y_test, probability)
        performance = metrics["f1_macro"]
        metric_name = "Macro-F1"

    return {
        "model": selected["model"],
        "threshold": selected["threshold"],
        "valid_accuracy": selected["valid_accuracy"],
        "test_performance": performance,
        "metric": metric_name,
    }


def make_figure(results: pd.DataFrame) -> None:
    colors = {"hitter": "#2F6B9A", "sp": "#D9822B", "rp": "#4E9F75"}
    labels = {"hitter": "Hitter", "sp": "Starting Pitcher", "rp": "Relief Pitcher"}
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), constrained_layout=True)
    panels = [
        ("2class", "agreement", "(a) Binary Label Agreement", "Agreement with Baseline"),
        ("3class", "agreement", "(b) Three-Class Label Agreement", "Agreement with Baseline"),
        ("2class", "test_performance", "(c) Binary Test Performance", "High-class F1"),
        ("3class", "test_performance", "(d) Three-Class Test Performance", "Macro-F1"),
    ]

    for ax, (task, value, title, ylabel) in zip(axes.flat, panels):
        subset = results[results["task"].eq(task)]
        for role in ("hitter", "sp", "rp"):
            role_data = subset[subset["role"].eq(role)].sort_values("war_weight")
            ax.plot(
                role_data["war_weight"],
                role_data[value],
                marker="o",
                linewidth=2,
                markersize=6,
                color=colors[role],
                label=labels[role],
            )
        ax.axvline(0.5, color="#666666", linestyle="--", linewidth=1.2)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("WAR Weight")
        ax.set_ylabel(ylabel)
        ax.set_xticks(WEIGHTS)
        ax.grid(axis="y", alpha=0.25)
        if value == "agreement":
            ax.set_ylim(0.75, 1.015)
        else:
            ax.set_ylim(bottom=0)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle(
        "Sensitivity Analysis of Alternative Composite Weights",
        fontsize=15, weight="bold"
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"figure_weight_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for role, cfg in base.ROLE_CONFIG.items():
        source = pd.read_csv(DATA_DIR / f"{role}_cohort_v2.csv")
        source["split"] = base.temporal_split(source)
        source = source[source["split"].notna()].copy()
        baseline = weighted_labels(source, cfg, 0.5)

        for weight in WEIGHTS:
            current = weighted_labels(source, cfg, weight)
            for task in ("2class", "3class"):
                agreement = float(
                    (current[f"label_{task}"] == baseline[f"label_{task}"]).mean()
                )
                fitted = fit_and_select(current, cfg, task)
                rows.append({
                    "role": role,
                    "task": task,
                    "war_weight": weight,
                    "utilization_weight": 1 - weight,
                    "agreement": agreement,
                    **fitted,
                })

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "weight_sensitivity_results.csv", index=False)
    make_figure(results)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
