"""
Create publication-ready model-contribution figures from the final
temporal-model specification.

Figure 5
--------
Hitter:
  (a) 2-class Logistic Regression coefficients
  (b) 3-class Random Forest impurity-based feature importance

Figure 6
--------
Pitchers:
  (a) SP 2-class Logistic Regression coefficients
  (b) SP 3-class Random Forest feature importance
  (c) RP 2-class Logistic Regression coefficients
  (d) RP 3-class Random Forest feature importance

This script uses the same temporal split, outcome construction, feature lists,
preprocessing, and estimator specifications as run_temporal_final_models.py.
No separate XGBoost interpretation model is fitted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.base import clone
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# This file is assumed to be:
# PROJECT_ROOT/src/plot_final_model_contributions.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.run_temporal_final_models import (
    DATA,
    ROLE_CONFIG,
    fit_outcomes,
    make_preprocessor,
    models,
    temporal_split,
)


OUTPUT = PROJECT_ROOT / "outputs" / "final_model_contributions"

TOP_N = 8
OUTCOME = "composite"

ROLE_LABELS = {
    "hitter": "Hitter",
    "sp": "Starting Pitcher",
    "rp": "Relief Pitcher",
}


# ---------------------------------------------------------------------
# Publication labels and colors
# ---------------------------------------------------------------------

FEATURE_LABELS = {
    "AAV": "Average annual value (AAV)",
    "Yrs": "Contract length",
    "age_at_contract": "Age at contract",
    "pre_WAR_3y_avg": "Mean WAR (previous 3 years)",
    "pre_WAR_3y_std": "WAR variability (previous 3 years)",
    "pre_PA_3y": "Plate appearances (previous 3 years)",
    "pre_OPSplus_3y_avg": "Mean OPS+ (previous 3 years)",
    "pre_BB_rate_3y_avg": "Mean walk rate (previous 3 years)",
    "pre_K_rate_3y_avg": "Mean strikeout rate (previous 3 years)",
    "pre_IP_3y": "Innings pitched (previous 3 years)",
    "pre_G_3y_sum": "Games pitched (previous 3 years)",
    "pre_GS_3y_sum": "Games started (previous 3 years)",
    "pre_ERA_3y_avg": "Mean ERA (previous 3 years)",
    "pre_FIP_3y_avg": "Mean FIP (previous 3 years)",
    "pre_KBB_3y_avg": "Mean K/BB (previous 3 years)",
    "Pos_group_C": "Position: Catcher",
    "Pos_group_DH": "Position: Designated hitter",
    "Pos_group_IF": "Position: Infielder",
    "Pos_group_OF": "Position: Outfielder",
}

POSITIVE_COLOR = "#2166AC"
NEGATIVE_COLOR = "#B2182B"
RF_COLOR = "#3D7F5F"
TEXT_COLOR = "#2F2F2F"
GRID_COLOR = "#B0B0B0"


def clean_feature_name(name: str) -> str:
    """Remove ColumnTransformer prefixes while preserving category labels."""
    for prefix in ("numeric__", "categorical__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def display_feature_name(name: str) -> str:
    """Convert internal variable names to publication-friendly labels."""
    return FEATURE_LABELS.get(name, name.replace("_", " "))


# ---------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------

def fit_role_models(role: str) -> tuple[Pipeline, Pipeline]:
    """
    Fit the specified 2-class Logistic Regression and
    3-class Random Forest pipelines for one player role.
    """
    cfg = ROLE_CONFIG[role]

    cohort_path = DATA / f"{role}_cohort_v2.csv"

    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Cohort file not found: {cohort_path.resolve()}"
        )

    frame = pd.read_csv(cohort_path)

    frame["split"] = temporal_split(frame)
    frame = frame[frame["split"].notna()].copy()

    frame, _ = fit_outcomes(frame, cfg)

    train = frame[frame["split"].eq("train")].copy()
    features = list(cfg["features"])

    if train.empty:
        raise ValueError(f"No training observations found for role: {role}")

    # 2-class Logistic Regression
    target_2class = f"{OUTCOME}_2class"

    logistic_estimator = models(
        "2class",
        train[target_2class],
    )["logistic"]

    logistic_pipeline = Pipeline([
        (
            "preprocessor",
            make_preprocessor(train[features]),
        ),
        (
            "model",
            clone(logistic_estimator),
        ),
    ])

    logistic_pipeline.fit(
        train[features],
        train[target_2class],
    )

    # 3-class Random Forest
    target_3class = f"{OUTCOME}_3class"

    rf_estimator = models(
        "3class",
        train[target_3class],
    )["rf"]

    rf_pipeline = Pipeline([
        (
            "preprocessor",
            make_preprocessor(train[features]),
        ),
        (
            "model",
            clone(rf_estimator),
        ),
    ])

    rf_pipeline.fit(
        train[features],
        train[target_3class],
    )

    return logistic_pipeline, rf_pipeline


# ---------------------------------------------------------------------
# Contribution extraction
# ---------------------------------------------------------------------

def extract_logistic_coefficients(
    pipeline: Pipeline,
    role: str,
) -> pd.DataFrame:
    """Extract signed coefficients from the fitted binary Logistic model."""
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]

    feature_names = [
        clean_feature_name(name)
        for name in preprocessor.get_feature_names_out()
    ]

    coefficients = np.asarray(estimator.coef_)

    if coefficients.shape[0] != 1:
        raise ValueError(
            "The 2-class Logistic model must contain one coefficient vector. "
            f"Observed shape: {coefficients.shape}"
        )

    if len(feature_names) != coefficients.shape[1]:
        raise ValueError(
            "Feature-name and coefficient lengths do not match: "
            f"{len(feature_names)} names vs. "
            f"{coefficients.shape[1]} coefficients."
        )

    result = pd.DataFrame({
        "role": role,
        "task": "2class",
        "model": "logistic",
        "feature": feature_names,
        "value": coefficients[0],
    })

    result["abs_value"] = result["value"].abs()

    return result.sort_values(
        "abs_value",
        ascending=False,
        ignore_index=True,
    )


def extract_rf_importance(
    pipeline: Pipeline,
    role: str,
) -> pd.DataFrame:
    """Extract impurity-based importance from the fitted multiclass RF."""
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]

    feature_names = [
        clean_feature_name(name)
        for name in preprocessor.get_feature_names_out()
    ]

    importances = np.asarray(estimator.feature_importances_)

    if len(feature_names) != len(importances):
        raise ValueError(
            "Feature-name and importance lengths do not match: "
            f"{len(feature_names)} names vs. "
            f"{len(importances)} importance values."
        )

    result = pd.DataFrame({
        "role": role,
        "task": "3class",
        "model": "rf",
        "feature": feature_names,
        "value": importances,
    })

    result["abs_value"] = result["value"]

    return result.sort_values(
        "value",
        ascending=False,
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# Figure styling
# ---------------------------------------------------------------------

def apply_publication_style() -> None:
    """Set consistent publication-oriented Matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    })


def style_axis(ax: plt.Axes) -> None:
    """Apply common formatting to one subplot."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    ax.tick_params(
        axis="both",
        colors=TEXT_COLOR,
        length=3,
    )

    ax.grid(
        axis="x",
        color=GRID_COLOR,
        linestyle="--",
        linewidth=0.6,
        alpha=0.45,
        zorder=0,
    )

    ax.set_axisbelow(True)


def add_panel_label(
    ax: plt.Axes,
    panel_label: str,
) -> None:
    """Place a panel label inside the upper-left corner."""
    ax.text(
        0.0,
        1.055,
        panel_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="black",
        ha="left",
        va="bottom",
    )


# ---------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------

def plot_logistic(
    ax: plt.Axes,
    frame: pd.DataFrame,
    panel_label: str,
    title: str,
) -> None:
    """Plot the largest absolute binary Logistic coefficients."""
    top = (
        frame.nlargest(TOP_N, "abs_value")
        .sort_values("value", ascending=True)
        .copy()
    )

    top["label"] = top["feature"].map(display_feature_name)

    colors = np.where(
        top["value"].ge(0),
        POSITIVE_COLOR,
        NEGATIVE_COLOR,
    )

    ax.barh(
        top["label"],
        top["value"],
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        height=0.68,
        zorder=3,
    )

    ax.axvline(
        0,
        color="#1F1F1F",
        linewidth=0.9,
        zorder=4,
    )

    maximum = max(float(top["value"].abs().max()), 0.01)
    limit = maximum * 1.30
    ax.set_xlim(-limit, limit)

    label_offset = maximum * 0.04

    for y_position, value in enumerate(top["value"]):
        if value >= 0:
            ax.text(
                value + label_offset,
                y_position,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=8,
                color=TEXT_COLOR,
            )
        else:
            ax.text(
                value - label_offset,
                y_position,
                f"{value:.2f}",
                va="center",
                ha="right",
                fontsize=8,
                color=TEXT_COLOR,
            )

    ax.set_title(
        title,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel("Logistic regression coefficient")

    add_panel_label(ax, panel_label)
    style_axis(ax)


def plot_rf(
    ax: plt.Axes,
    frame: pd.DataFrame,
    panel_label: str,
    title: str,
) -> None:
    """Plot the largest multiclass Random Forest feature importances."""
    top = (
        frame.nlargest(TOP_N, "value")
        .sort_values("value", ascending=True)
        .copy()
    )

    top["label"] = top["feature"].map(display_feature_name)

    ax.barh(
        top["label"],
        top["value"],
        color=RF_COLOR,
        edgecolor="white",
        linewidth=0.7,
        height=0.68,
        zorder=3,
    )

    maximum = max(float(top["value"].max()), 0.01)
    ax.set_xlim(0, maximum * 1.30)

    for y_position, value in enumerate(top["value"]):
        ax.text(
            value + maximum * 0.025,
            y_position,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color=TEXT_COLOR,
        )

    ax.set_title(
        title,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel("Impurity-based feature importance")

    add_panel_label(ax, panel_label)
    style_axis(ax)


# ---------------------------------------------------------------------
# Figure creation and saving
# ---------------------------------------------------------------------

def save_hitter_figure(
    logistic: pd.DataFrame,
    rf: pd.DataFrame,
) -> None:
    """Create and save publication-ready Figure 5."""
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 6.3),
    )

    plot_logistic(
        axes[0],
        logistic,
        "(a)",
        "2-Class Logistic Regression",
    )

    plot_rf(
        axes[1],
        rf,
        "(b)",
        "3-Class Random Forest",
    )

    legend_items = [
        Patch(
            facecolor=POSITIVE_COLOR,
            label="Positive coefficient",
        ),
        Patch(
            facecolor=NEGATIVE_COLOR,
            label="Negative coefficient",
        ),
        Patch(
            facecolor=RF_COLOR,
            label="RF feature importance",
        ),
    ]

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=9,
    )

    fig.suptitle(
        "Model-Based Variable Contributions for Hitters",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        top=0.86,
        bottom=0.17,
        wspace=0.45,
    )

    png_path = OUTPUT / "figure5_hitter_model_contributions.png"
    pdf_path = OUTPUT / "figure5_hitter_model_contributions.pdf"

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"\nFigure 5 PNG saved: {png_path.resolve()}")
    print(f"Figure 5 PDF saved: {pdf_path.resolve()}")
    print(f"PNG exists: {png_path.exists()}")
    print(f"PDF exists: {pdf_path.exists()}")


def save_pitcher_figure(
    results: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """Create and save publication-ready Figure 6."""
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15.5, 12.5),
    )

    plot_logistic(
        axes[0, 0],
        results["sp"]["logistic"],
        "(a)",
        "Starting Pitcher: 2-Class Logistic Regression",
    )

    plot_rf(
        axes[0, 1],
        results["sp"]["rf"],
        "(b)",
        "Starting Pitcher: 3-Class Random Forest",
    )

    plot_logistic(
        axes[1, 0],
        results["rp"]["logistic"],
        "(c)",
        "Relief Pitcher: 2-Class Logistic Regression",
    )

    plot_rf(
        axes[1, 1],
        results["rp"]["rf"],
        "(d)",
        "Relief Pitcher: 3-Class Random Forest",
    )

    legend_items = [
        Patch(
            facecolor=POSITIVE_COLOR,
            label="Positive coefficient",
        ),
        Patch(
            facecolor=NEGATIVE_COLOR,
            label="Negative coefficient",
        ),
        Patch(
            facecolor=RF_COLOR,
            label="RF feature importance",
        ),
    ]

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=9,
    )

    fig.suptitle(
        "Model-Based Variable Contributions for Pitchers",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        top=0.91,
        bottom=0.09,
        wspace=0.45,
        hspace=0.42,
    )

    png_path = OUTPUT / "figure6_pitcher_model_contributions.png"
    pdf_path = OUTPUT / "figure6_pitcher_model_contributions.pdf"

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"\nFigure 6 PNG saved: {png_path.resolve()}")
    print(f"Figure 6 PDF saved: {pdf_path.resolve()}")
    print(f"PNG exists: {png_path.exists()}")
    print(f"PDF exists: {pdf_path.exists()}")


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def run() -> None:
    """Fit models, export contribution tables, and create Figures 5 and 6."""
    apply_publication_style()

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Project root: {PROJECT_ROOT.resolve()}")
    print(f"Data directory: {Path(DATA).resolve()}")
    print(f"Output directory: {OUTPUT.resolve()}")

    role_results: dict[str, dict[str, pd.DataFrame]] = {}
    all_results: list[pd.DataFrame] = []

    for role in ("hitter", "sp", "rp"):
        print(f"\nFitting contribution models for: {ROLE_LABELS[role]}")

        logistic_pipeline, rf_pipeline = fit_role_models(role)

        logistic = extract_logistic_coefficients(
            logistic_pipeline,
            role,
        )

        rf = extract_rf_importance(
            rf_pipeline,
            role,
        )

        role_results[role] = {
            "logistic": logistic,
            "rf": rf,
        }

        all_results.extend([
            logistic,
            rf,
        ])

        logistic_csv = (
            OUTPUT / f"{role}_2class_logistic_coefficients.csv"
        )

        rf_csv = (
            OUTPUT / f"{role}_3class_rf_importance.csv"
        )

        logistic.to_csv(
            logistic_csv,
            index=False,
            encoding="utf-8-sig",
        )

        rf.to_csv(
            rf_csv,
            index=False,
            encoding="utf-8-sig",
        )

    combined_csv = OUTPUT / "all_model_contributions.csv"

    pd.concat(
        all_results,
        ignore_index=True,
    ).to_csv(
        combined_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # Figure 5: Hitter
    save_hitter_figure(
        role_results["hitter"]["logistic"],
        role_results["hitter"]["rf"],
    )

    # Figure 6: Starting and relief pitchers
    save_pitcher_figure(role_results)

    # Print top contributions
    for role in ("hitter", "sp", "rp"):
        print(f"\n[{ROLE_LABELS[role]} | 2-class Logistic]")

        print(
            role_results[role]["logistic"]
            .head(TOP_N)[["feature", "value"]]
            .to_string(index=False)
        )

        print(f"\n[{ROLE_LABELS[role]} | 3-class Random Forest]")

        print(
            role_results[role]["rf"]
            .head(TOP_N)[["feature", "value"]]
            .to_string(index=False)
        )

    print("\nAll outputs were created successfully.")
    print(f"Output directory: {OUTPUT.resolve()}")


if __name__ == "__main__":
    run()