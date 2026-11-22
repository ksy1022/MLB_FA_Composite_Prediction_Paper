"""Build paper-ready tables and figures for the revised optimized 2-class analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "optimized_2class"
PAPER_OUT = OUT / "paper"
ROLE_ORDER = ["hitter", "sp", "rp"]
ROLE_LABELS = {
    "hitter": "Hitter",
    "sp": "Starting Pitcher",
    "rp": "Relief Pitcher",
}


def main() -> None:
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    search_path = OUT / "cutoff_search_validation.csv"
    config_path = OUT / "selected_role_config.json"
    metrics_path = OUT / "final_test_metrics.csv"
    preds_path = OUT / "final_predictions.csv"
    for p in (search_path, config_path, metrics_path, preds_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required output: {p}")

    search = pd.read_csv(search_path)
    final = pd.read_csv(metrics_path)
    pred = pd.read_csv(preds_path, dtype={"player_key": str})
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------
    # Table: role-specific cutoff/model configuration
    # ------------------------------------------------------------
    rows = []
    for role in ROLE_ORDER:
        role_cfg = config["roles"][role]
        for outcome in ("composite", "war"):
            oc = role_cfg["outcomes"][outcome]
            rows.append(
                {
                    "Role": ROLE_LABELS[role],
                    "Outcome": "Composite" if outcome == "composite" else "WAR-only",
                    "Selected High cutoff": f"Top {role_cfg['selected_top_percent']:.0f}%",
                    "Selected model": oc["model"],
                    "Probability threshold": oc["probability_threshold"],
                    "Validation F1": oc["validation_metrics"]["f1"],
                    "Validation balanced accuracy": oc["validation_metrics"]["balanced_accuracy"],
                    "Test Accuracy": oc["test_metrics"]["accuracy"],
                    "Test F1": oc["test_metrics"]["f1"],
                    "Test PR-AUC": oc["test_metrics"]["pr_auc"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(PAPER_OUT / "table_role_cutoff_model_selection.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Figure: cutoff-selection curve using the best model at each cutoff.
    # This is a development/selection figure, not a test-performance figure.
    # ------------------------------------------------------------
    best_per_cutoff = (
        search.sort_values(
            ["role", "top_ratio", "f1", "balanced_accuracy", "pr_auc", "recall"],
            ascending=[True, True, False, False, False, False],
            kind="mergesort",
        )
        .groupby(["role", "top_ratio"], as_index=False)
        .head(1)
    )
    best_per_cutoff.to_csv(
        PAPER_OUT / "cutoff_selection_curve_data.csv", index=False, encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for role in ROLE_ORDER:
        part = best_per_cutoff[best_per_cutoff["role"].eq(role)].sort_values("top_ratio")
        ax.plot(part["top_ratio"] * 100, part["f1"], marker="o", label=ROLE_LABELS[role])
        selected = float(config["roles"][role]["selected_top_ratio"])
        selected_row = part[np.isclose(part["top_ratio"], selected)].iloc[0]
        ax.scatter([selected * 100], [selected_row["f1"]], s=90, zorder=4)
        ax.annotate(
            f"{selected:.0%}",
            (selected * 100, selected_row["f1"]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("High group cutoff: top proportion (%)")
    ax.set_ylabel("Validation High-class F1")
    ax.set_title("Role-specific cutoff selection on validation data")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PAPER_OUT / "figure_cutoff_selection_validation.png", dpi=300)
    fig.savefig(PAPER_OUT / "figure_cutoff_selection_validation.pdf")
    plt.close(fig)

    # ------------------------------------------------------------
    # Figure: final Composite precision-recall curves on test set.
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    pr_rows = []
    for role in ROLE_ORDER:
        part = pred[
            pred["role"].eq(role)
            & pred["outcome"].eq("composite")
            & pred["split"].eq("test")
        ].copy()
        precision, recall, thresholds = precision_recall_curve(
            part["y_true"], part["probability_high"]
        )
        test_metric = final[
            final["role"].eq(role) & final["outcome"].eq("composite")
        ].iloc[0]
        label = f"{ROLE_LABELS[role]} (AP={test_metric['pr_auc']:.3f})"
        ax.plot(recall, precision, label=label)
        baseline = float(part["y_true"].mean())
        pr_rows.append({"role": role, "test_high_rate": baseline, "pr_auc": test_metric["pr_auc"]})
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves for Final Composite Binary Models")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PAPER_OUT / "figure_pr_curves_final_composite.png", dpi=300)
    fig.savefig(PAPER_OUT / "figure_pr_curves_final_composite.pdf")
    plt.close(fig)
    pd.DataFrame(pr_rows).to_csv(PAPER_OUT / "pr_curve_summary.csv", index=False, encoding="utf-8-sig")

    # A compact final test table for the paper.
    final_table = final.copy()
    final_table["Role"] = final_table["role"].map(ROLE_LABELS)
    final_table["Outcome"] = final_table["outcome"].map({"composite": "Composite", "war": "WAR-only"})
    keep = [
        "Role", "Outcome", "selected_top_ratio", "selected_model", "threshold",
        "accuracy", "balanced_accuracy", "precision", "recall", "f1", "pr_auc",
    ]
    final_table[keep].to_csv(PAPER_OUT / "table_final_2class_test_performance.csv", index=False, encoding="utf-8-sig")

    print(table.to_string(index=False))
    print(f"\nPaper outputs saved to: {PAPER_OUT}")


if __name__ == "__main__":
    main()
