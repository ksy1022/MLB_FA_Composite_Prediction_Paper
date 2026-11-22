from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Path settings
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "outputs" / "paper"

INPUT_FILE = (
    OUTPUT_DIR
    / "table8_sensitivity_recomputed.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "figure9_construct_validity_weight_sensitivity.png"
)


# ============================================================
# 2. Role labels
# ============================================================

ROLE_SHORT = {
    "Hitter": "Hitter",
    "Starting Pitcher": "SP",
    "Relief Pitcher": "RP",
}


# ============================================================
# 3. Helper functions
# ============================================================

def clean_axes(ax):
    """논문용으로 불필요한 테두리를 제거한다."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)


def add_bar_labels(
    ax,
    bars,
    offset: float,
    fontsize: int = 9,
):
    """막대 위에 값을 표시한다."""

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# ============================================================
# 4. Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "\nSensitivity-analysis result not found.\n\n"
            "Run first:\n"
            "python src/paper/sensitivity_analysis.py\n"
        )

    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = [
        "Role",
        "PC1 Variance Explained",
        "WAR Loading",
        "Utilization Loading",
        "2-class Label Agreement (0.4:0.6)",
        "2-class Label Agreement (0.6:0.4)",
        "3-class Label Agreement (0.4:0.6)",
        "3-class Label Agreement (0.6:0.4)",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Required columns are missing:\n{missing}"
        )

    df["Short Role"] = (
        df["Role"]
        .map(ROLE_SHORT)
    )

    # --------------------------------------------------------
    # Plot settings
    # --------------------------------------------------------

    x = np.arange(len(df))
    width = 0.33

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.4),
    )

    # ========================================================
    # (a) PC1 Loadings
    # ========================================================

    ax1 = axes[0]

    bars_war = ax1.bar(
        x - width / 2,
        df["WAR Loading"],
        width,
        label="WAR",
    )

    bars_util = ax1.bar(
        x + width / 2,
        df["Utilization Loading"],
        width,
        label="Utilization",
    )

    ax1.set_title(
        "(a) PC1 Loadings",
        fontsize=12,
        fontweight="semibold",
        pad=12,
    )

    ax1.set_xticks(x)

    ax1.set_xticklabels(
        df["Short Role"],
        fontsize=10,
    )

    ax1.set_ylabel(
        "Loading",
        fontsize=10,
    )

    ax1.set_ylim(
        0,
        0.82,
    )

    clean_axes(ax1)

    # ========================================================
    # (b) 2-Class Label Agreement
    # ========================================================

    ax2 = axes[1]

    bars_2_04 = ax2.bar(
        x - width / 2,
        df[
            "2-class Label Agreement (0.4:0.6)"
        ],
        width,
        label="WAR:Util. = 0.4:0.6",
    )

    bars_2_06 = ax2.bar(
        x + width / 2,
        df[
            "2-class Label Agreement (0.6:0.4)"
        ],
        width,
        label="WAR:Util. = 0.6:0.4",
    )

    ax2.set_title(
        "(b) 2-Class Label Agreement",
        fontsize=12,
        fontweight="semibold",
        pad=12,
    )

    ax2.set_xticks(x)

    ax2.set_xticklabels(
        df["Short Role"],
        fontsize=10,
    )

    ax2.set_ylabel(
        "Agreement",
        fontsize=10,
    )

    ax2.set_ylim(
        0.90,
        1.005,
    )

    clean_axes(ax2)

    add_bar_labels(
        ax2,
        bars_2_04,
        0.0015,
    )

    add_bar_labels(
        ax2,
        bars_2_06,
        0.0015,
    )

    # ========================================================
    # (c) 3-Class Label Agreement
    # ========================================================

    ax3 = axes[2]

    bars_3_04 = ax3.bar(
        x - width / 2,
        df[
            "3-class Label Agreement (0.4:0.6)"
        ],
        width,
        label="WAR:Util. = 0.4:0.6",
    )

    bars_3_06 = ax3.bar(
        x + width / 2,
        df[
            "3-class Label Agreement (0.6:0.4)"
        ],
        width,
        label="WAR:Util. = 0.6:0.4",
    )

    ax3.set_title(
        "(c) 3-Class Label Agreement",
        fontsize=12,
        fontweight="semibold",
        pad=12,
    )

    ax3.set_xticks(x)

    ax3.set_xticklabels(
        df["Short Role"],
        fontsize=10,
    )

    ax3.set_ylabel(
        "Agreement",
        fontsize=10,
    )

    ax3.set_ylim(
        0.90,
        1.005,
    )

    clean_axes(ax3)

    add_bar_labels(
        ax3,
        bars_3_04,
        0.0015,
    )

    add_bar_labels(
        ax3,
        bars_3_06,
        0.0015,
    )

    # ========================================================
    # Bottom row
    # ========================================================

    # 핵심:
    # 두 범례의 y 좌표를 완전히 동일하게 사용한다.
    LEGEND_Y = 0.105
    NOTE_Y = 0.045

    # --------------------------------------------------------
    # Left legend
    # --------------------------------------------------------

    fig.legend(
        handles=[
            bars_war,
            bars_util,
        ],
        labels=[
            "WAR",
            "Utilization",
        ],
        loc="center",
        bbox_to_anchor=(
            0.185,
            LEGEND_Y,
        ),
        frameon=False,
        fontsize=9,
        ncol=2,
        columnspacing=1.8,
        handletextpad=0.7,
    )

    # --------------------------------------------------------
    # PC1 variance note
    # --------------------------------------------------------

    variance_text = (
        "PC1 variance explained: "
        f"Hitter {df.loc[df['Role'] == 'Hitter', 'PC1 Variance Explained'].iloc[0]:.3f}, "
        f"SP {df.loc[df['Role'] == 'Starting Pitcher', 'PC1 Variance Explained'].iloc[0]:.3f}, "
        f"RP {df.loc[df['Role'] == 'Relief Pitcher', 'PC1 Variance Explained'].iloc[0]:.3f}"
    )

    fig.text(
        0.185,
        NOTE_Y,
        variance_text,
        ha="center",
        va="center",
        fontsize=8.5,
    )

    # --------------------------------------------------------
    # Shared weight legend
    # --------------------------------------------------------

    weight_handles = [
        bars_2_04,
        bars_2_06,
    ]

    weight_labels = [
        "WAR:Util. = 0.4:0.6",
        "WAR:Util. = 0.6:0.4",
    ]

    fig.legend(
        handles=weight_handles,
        labels=weight_labels,
        loc="center",
        bbox_to_anchor=(
            0.705,
            LEGEND_Y,
        ),
        frameon=False,
        fontsize=9,
        ncol=2,
        columnspacing=1.8,
        handletextpad=0.7,
    )

    # ========================================================
    # Layout
    # ========================================================

    plt.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.90,
        bottom=0.22,
        wspace=0.32,
    )

    # ========================================================
    # Save
    # ========================================================

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("=" * 80)
    print("Figure 9 generated successfully.")
    print("=" * 80)
    print(f"\nSaved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()