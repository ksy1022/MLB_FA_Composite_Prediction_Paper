"""Generate the paper-ready Composite Score distribution figure in English."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed_v2"
OUTPUT_DIR = ROOT / "outputs" / "paper_figures"
ROLE_CONFIG = {
    "hitter": {
        "file": "hitter_cohort_v2.csv",
        "name": "Hitters",
        "playtime": "post_PA_3y",
        "top_ratio": 0.40,
        "color": "#4C78A8",
    },
    "sp": {
        "file": "sp_cohort_v2.csv",
        "name": "Starting Pitchers",
        "playtime": "post_IP_3y",
        "top_ratio": 0.20,
        "color": "#F28E2B",
    },
    "rp": {
        "file": "rp_cohort_v2.csv",
        "name": "Relief Pitchers",
        "playtime": "post_IP_3y",
        "top_ratio": 0.20,
        "color": "#59A14F",
    },
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )


def make_training_scores(role: str, config: dict) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(DATA_DIR / config["file"])
    frame["fa_year"] = pd.to_numeric(frame["fa_year"], errors="coerce")
    train = frame.loc[frame["fa_year"].between(2012, 2017)].copy()

    result = {}
    for column, short in (
        ("post_WAR_3y_sum", "war"),
        (config["playtime"], "playtime"),
    ):
        median = float(train[column].median())
        filled = train[column].fillna(median)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        train[f"z_{short}"] = (filled - mean) / std
        result.update(
            {
                f"{short}_median": median,
                f"{short}_mean": mean,
                f"{short}_std": std,
            }
        )

    # 논문 수식과 동일한 동일가중 정의
    train["composite_score"] = (
        0.5 * train["z_war"] + 0.5 * train["z_playtime"]
    )
    cut = float(
        train["composite_score"].quantile(1.0 - config["top_ratio"])
    )
    result.update(
        {
            "role": role,
            "role_name": config["name"],
            "n": len(train),
            "min_year": int(train["fa_year"].min()),
            "max_year": int(train["fa_year"].max()),
            "success_top_ratio": config["top_ratio"],
            "success_cut": cut,
            "mean": float(train["composite_score"].mean()),
            "std": float(train["composite_score"].std(ddof=0)),
            "skewness": float(train["composite_score"].skew()),
            "minimum": float(train["composite_score"].min()),
            "maximum": float(train["composite_score"].max()),
        }
    )
    return train, result


def main() -> None:
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {}
    summaries = []
    for role, config in ROLE_CONFIG.items():
        datasets[role], summary = make_training_scores(role, config)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        OUTPUT_DIR / "figure1_composite_distribution_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_scores = pd.concat(
        [frame["composite_score"] for frame in datasets.values()],
        ignore_index=True,
    )
    lower, upper = all_scores.quantile([0.005, 0.995])
    padding = (upper - lower) * 0.06
    x_min, x_max = lower - padding, upper + padding
    bins = np.linspace(x_min, x_max, 27)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.15), sharex=True)

    for ax, (role, config), summary in zip(
        axes, ROLE_CONFIG.items(), summaries
    ):
        scores = datasets[role]["composite_score"]
        visible = scores[(scores >= x_min) & (scores <= x_max)]
        ax.hist(
            visible,
            bins=bins,
            color=config["color"],
            alpha=0.84,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.axvline(
            summary["success_cut"],
            color="#B22222",
            linestyle="--",
            linewidth=1.7,
            label=f"2-class cutoff: {summary['success_cut']:.3f}",
        )
        ax.set_title(
            f"{config['name']} (n={summary['n']})",
            fontsize=12,
            pad=9,
        )
        ax.set_xlabel("Composite Score")
        ax.grid(axis="y", linestyle=":", alpha=0.32)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=False, fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Frequency")
    fig.tight_layout()

    for extension, kwargs in {
        "png": {"dpi": 400},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(
            OUTPUT_DIR / f"figure1_composite_distribution_en.{extension}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)

    print(summary_df.to_string(index=False))
    print(f"\n저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()