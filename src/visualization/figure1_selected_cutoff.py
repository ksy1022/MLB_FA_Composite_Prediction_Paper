"""Regenerate Composite Score distribution figure using selected role cutoffs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELING = ROOT / "src" / "modeling"
sys.path.insert(0, str(MODELING))
from optimized_2class_common import (  # noqa: E402
    ROLE_CONFIG,
    ROLE_LABELS,
    add_binary_labels,
    load_role_frame,
    prepare_scores,
)

OUT = ROOT / "outputs" / "optimized_2class"
PLOT_OUT = OUT / "paper"
CONFIG_PATH = OUT / "selected_role_config.json"


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Run run_optimized_2class.py first: {CONFIG_PATH}")
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    selected = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    frames = {}
    summaries = []
    for role, cfg in ROLE_CONFIG.items():
        raw = load_role_frame(role)
        scored, _ = prepare_scores(raw, cfg)
        top_ratio = float(selected["roles"][role]["selected_top_ratio"])
        labeled, cuts = add_binary_labels(scored, cfg, top_ratio)
        train = labeled[labeled["split"].eq("train")].copy()
        frames[role] = train
        summaries.append(
            {
                "role": role,
                "role_label": ROLE_LABELS[role],
                "train_n": len(train),
                "selected_top_ratio": top_ratio,
                "composite_cut": cuts["composite_cut"],
                "score_mean": train["composite_score"].mean(),
                "score_std": train["composite_score"].std(ddof=0),
                "score_skewness": train["composite_score"].skew(),
            }
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(PLOT_OUT / "figure1_selected_cutoff_summary.csv", index=False, encoding="utf-8-sig")

    all_scores = pd.concat([x["composite_score"] for x in frames.values()], ignore_index=True)
    lower, upper = all_scores.quantile([0.005, 0.995])
    padding = float((upper - lower) * 0.06)
    xmin, xmax = float(lower - padding), float(upper + padding)
    bins = np.linspace(xmin, xmax, 27)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.15), sharex=True)
    for ax, role in zip(axes, ("hitter", "sp", "rp")):
        row = summary[summary["role"].eq(role)].iloc[0]
        score = frames[role]["composite_score"]
        visible = score[(score >= xmin) & (score <= xmax)]
        ax.hist(visible, bins=bins, alpha=0.84, edgecolor="white", linewidth=0.6)
        ax.axvline(
            row["composite_cut"],
            linestyle="--",
            linewidth=1.7,
            label=f"Top {row['selected_top_ratio']:.0%} cutoff",
        )
        ax.set_title(f"{ROLE_LABELS[role]} (train n={int(row['train_n'])})")
        ax.set_xlabel("Composite Score")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(PLOT_OUT / "figure1_composite_distribution_selected_cutoff.png", dpi=300)
    fig.savefig(PLOT_OUT / "figure1_composite_distribution_selected_cutoff.pdf")
    plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
