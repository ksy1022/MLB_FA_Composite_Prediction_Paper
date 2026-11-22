"""Regenerate 2-class variable-contribution outputs for the newly selected models.

Because cutoff reselection can change the selected model family, this script is
model-aware:
- Logistic Regression -> signed coefficients
- Random Forest / XGBoost -> feature importance

The selected model is read from outputs/optimized_2class/selected_role_config.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELING = ROOT / "src" / "modeling"
sys.path.insert(0, str(MODELING))
from optimized_2class_common import (  # noqa: E402
    ROLE_CONFIG,
    ROLE_LABELS,
    add_binary_labels,
    candidate_models,
    fit_pipeline,
    load_role_frame,
    prepare_scores,
)

OUT = ROOT / "outputs" / "optimized_2class"
CONTRIB_OUT = OUT / "contributions"
CONFIG_PATH = OUT / "selected_role_config.json"
TOP_N = 10


def clean_feature_name(name: str) -> str:
    for prefix in ("numeric__", "categorical__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def extract_contribution(pipe, role: str, model_name: str) -> pd.DataFrame:
    preprocessor = pipe.named_steps["preprocessor"]
    estimator = pipe.named_steps["model"]
    feature_names = [clean_feature_name(x) for x in preprocessor.get_feature_names_out()]

    if model_name == "logistic":
        values = estimator.coef_[0]
        kind = "coefficient"
        signed = True
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        kind = "feature_importance"
        signed = False
    else:
        raise TypeError(f"Unsupported selected model for contribution extraction: {model_name}")

    frame = pd.DataFrame(
        {
            "role": role,
            "role_label": ROLE_LABELS[role],
            "model": model_name,
            "kind": kind,
            "feature": feature_names,
            "value": values,
        }
    )
    frame["abs_value"] = frame["value"].abs() if signed else frame["value"]
    return frame.sort_values("abs_value", ascending=False, ignore_index=True)


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Run run_optimized_2class.py first: {CONFIG_PATH}")
    selected = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    CONTRIB_OUT.mkdir(parents=True, exist_ok=True)

    role_frames: dict[str, pd.DataFrame] = {}
    for role, cfg in ROLE_CONFIG.items():
        raw = load_role_frame(role)
        scored, _ = prepare_scores(raw, cfg)
        top_ratio = float(selected["roles"][role]["selected_top_ratio"])
        labeled, _ = add_binary_labels(scored, cfg, top_ratio)
        train = labeled[labeled["split"].eq("train")].copy()
        target = "composite_2class"
        model_name = selected["roles"][role]["outcomes"]["composite"]["model"]
        estimator = candidate_models(train[target])[model_name]
        pipe = fit_pipeline(train, cfg["features"], target, estimator)
        contrib = extract_contribution(pipe, role, model_name)
        role_frames[role] = contrib
        contrib.to_csv(
            CONTRIB_OUT / f"{role}_selected_2class_contributions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.concat(role_frames.values(), ignore_index=True).to_csv(
        CONTRIB_OUT / "all_selected_2class_contributions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, role in zip(axes, ("hitter", "sp", "rp")):
        frame = role_frames[role].head(TOP_N).sort_values("value")
        model_name = frame["model"].iloc[0]
        kind = frame["kind"].iloc[0]
        if kind == "coefficient":
            ax.barh(frame["feature"], frame["value"])
            ax.axvline(0, linewidth=0.8)
            ax.set_xlabel("Coefficient")
        else:
            ax.barh(frame["feature"], frame["value"])
            ax.set_xlabel("Feature importance")
        ax.set_title(f"{ROLE_LABELS[role]}: {model_name}")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Variable Contributions of Final Composite 2-Class Models")
    fig.tight_layout()
    fig.savefig(CONTRIB_OUT / "figure_selected_2class_contributions.png", dpi=300)
    fig.savefig(CONTRIB_OUT / "figure_selected_2class_contributions.pdf")
    plt.close(fig)

    for role, frame in role_frames.items():
        print(f"\n[{ROLE_LABELS[role]}]")
        print(frame.head(TOP_N)[["feature", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
