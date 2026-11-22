"""Check whether training Composite Score has a defensible natural 2-group structure.

Purpose
-------
This script does NOT choose a percentile cutoff by prediction performance.
It asks a different question:

    "Does the 2012-2017 training Composite Score distribution itself contain
     a stable, genuinely bimodal 2-group structure?"

For each role (hitter / sp / rp), it:

1. Uses only 2012-2017 contracts.
2. Standardizes post-contract WAR and PA/IP using training data only.
3. Builds Composite = 0.5*z(WAR) + 0.5*z(Utilization).
4. Fits 1-, 2-, and 3-component Gaussian mixture models.
5. Compares BIC/AIC.
6. Checks whether the fitted 2-component density is actually bimodal.
   (A 2-component GMM can fit a skewed unimodal distribution, so BIC alone
   is not enough evidence of two natural groups.)
7. Computes Ashman's D separation for the 2-component fit.
8. Finds the equal-posterior intersection between the two components, if
   one exists between the component means.
9. Performs player-cluster bootstrap stability analysis.
10. Saves CSV summaries, bootstrap draws, plots, and a short Markdown report.

Important
---------
The automatic "clean_2group_support" flag is a conservative decision aid,
not a formal statistical theorem. It requires:

- BIC favors 2 over 1 by at least 10,
- 2 components are the best BIC solution among k=1,2,3,
- the fitted 2-component mixture has two visible modes,
- Ashman's D >= 2,
- a posterior-equality boundary exists between component means,
- bootstrap support for 2-vs-1 BIC preference >= 80%,
- bootstrap support for a genuinely bimodal 2-component density >= 70%.

If these conditions are not met, do not claim that the data reveal a
"natural optimal binary cutoff".
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed_v2"
OUT_DIR = ROOT / "outputs" / "composite_distribution_structure"

RANDOM_SEED = 42
N_BOOTSTRAP = 200
GMM_N_INIT = 10
GMM_MAX_ITER = 2000
REG_COVAR = 1e-6

ROLE_CONFIG = {
    "hitter": {
        "label": "Hitter",
        "file": "hitter_cohort_v2.csv",
        "war": "post_WAR_3y_sum",
        "utilization": "post_PA_3y",
    },
    "sp": {
        "label": "Starting Pitcher",
        "file": "sp_cohort_v2.csv",
        "war": "post_WAR_3y_sum",
        "utilization": "post_IP_3y",
    },
    "rp": {
        "label": "Relief Pitcher",
        "file": "rp_cohort_v2.csv",
        "war": "post_WAR_3y_sum",
        "utilization": "post_IP_3y",
    },
}


def load_training_composite(role: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    path = DATA_DIR / cfg["file"]
    if not path.exists():
        raise FileNotFoundError(f"Missing cohort file: {path}")

    df = pd.read_csv(path)
    required = {"fa_year", cfg["war"], cfg["utilization"]}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path.name} missing columns: {sorted(missing)}")

    df["fa_year"] = pd.to_numeric(df["fa_year"], errors="coerce")
    train = df.loc[df["fa_year"].between(2012, 2017)].copy()

    if train.empty:
        raise ValueError(f"{role}: no 2012-2017 training rows")

    params = {}
    for col, short in ((cfg["war"], "war"), (cfg["utilization"], "utilization")):
        values = pd.to_numeric(train[col], errors="coerce")
        median = float(values.median())
        filled = values.fillna(median)
        mean = float(filled.mean())
        sd = float(filled.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"{role}: invalid training SD for {col}: {sd}")
        train[f"z_{short}"] = (filled - mean) / sd
        params.update(
            {
                f"{short}_median": median,
                f"{short}_mean": mean,
                f"{short}_sd": sd,
            }
        )

    train["composite_score"] = (
        0.5 * train["z_war"] + 0.5 * train["z_utilization"]
    )

    params.update(
        {
            "n_contracts": int(len(train)),
            "n_players": int(train["player_key"].nunique())
            if "player_key" in train.columns
            else int(len(train)),
            "min_year": int(train["fa_year"].min()),
            "max_year": int(train["fa_year"].max()),
            "composite_mean": float(train["composite_score"].mean()),
            "composite_sd": float(train["composite_score"].std(ddof=0)),
            "composite_skew": float(train["composite_score"].skew()),
        }
    )
    return train, params


def fit_gmm(x: np.ndarray, k: int, seed: int = RANDOM_SEED) -> GaussianMixture:
    model = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=GMM_N_INIT,
        max_iter=GMM_MAX_ITER,
        reg_covar=REG_COVAR,
        random_state=seed,
    )
    model.fit(x.reshape(-1, 1))
    return model


def sorted_two_component_params(model: GaussianMixture) -> dict:
    if model.n_components != 2:
        raise ValueError("Expected a two-component GMM")

    means = model.means_.reshape(-1)
    variances = model.covariances_.reshape(-1)
    weights = model.weights_.reshape(-1)

    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    weights = weights[order]
    sds = np.sqrt(variances)

    ashman_d = (
        math.sqrt(2.0)
        * abs(means[1] - means[0])
        / math.sqrt(variances[0] + variances[1])
    )

    return {
        "low_mean": float(means[0]),
        "high_mean": float(means[1]),
        "low_sd": float(sds[0]),
        "high_sd": float(sds[1]),
        "low_weight": float(weights[0]),
        "high_weight": float(weights[1]),
        "ashman_d": float(ashman_d),
    }


def component_log_weighted_density(
    x: float, weight: float, mean: float, sd: float
) -> float:
    return math.log(weight) + norm.logpdf(x, loc=mean, scale=sd)


def posterior_boundary_between_means(params: dict) -> float | None:
    m1, m2 = params["low_mean"], params["high_mean"]
    w1, w2 = params["low_weight"], params["high_weight"]
    s1, s2 = params["low_sd"], params["high_sd"]

    def f(value: float) -> float:
        return component_log_weighted_density(value, w1, m1, s1) - (
            component_log_weighted_density(value, w2, m2, s2)
        )

    grid = np.linspace(m1, m2, 1001)
    vals = np.array([f(v) for v in grid])

    exact = np.flatnonzero(np.isclose(vals, 0.0, atol=1e-12))
    if len(exact):
        return float(grid[exact[len(exact) // 2]])

    changes = np.flatnonzero(np.signbit(vals[:-1]) != np.signbit(vals[1:]))
    if len(changes) == 0:
        return None

    midpoint = (m1 + m2) / 2
    roots = []
    for i in changes:
        try:
            root = brentq(f, float(grid[i]), float(grid[i + 1]))
            roots.append(float(root))
        except ValueError:
            continue

    if not roots:
        return None
    return min(roots, key=lambda r: abs(r - midpoint))


def mixture_pdf(model: GaussianMixture, grid: np.ndarray) -> np.ndarray:
    return np.exp(model.score_samples(grid.reshape(-1, 1)))


def count_visible_modes(model: GaussianMixture, x: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    sd = float(np.std(x, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0

    lo = float(np.min(x) - 1.5 * sd)
    hi = float(np.max(x) + 1.5 * sd)
    grid = np.linspace(lo, hi, 5000)
    density = mixture_pdf(model, grid)

    # Ignore tiny numerical shoulders.
    prominence = max(float(density.max()) * 0.02, 1e-12)
    peaks, _ = find_peaks(density, prominence=prominence)
    return int(len(peaks)), grid, density


def evaluate_distribution(x: np.ndarray, seed: int = RANDOM_SEED) -> tuple[dict, dict[int, GaussianMixture]]:
    models = {k: fit_gmm(x, k, seed=seed) for k in (1, 2, 3)}
    bics = {k: float(models[k].bic(x.reshape(-1, 1))) for k in models}
    aics = {k: float(models[k].aic(x.reshape(-1, 1))) for k in models}

    p2 = sorted_two_component_params(models[2])
    boundary = posterior_boundary_between_means(p2)
    mode_count, _, _ = count_visible_modes(models[2], x)

    best_k_bic = min(bics, key=bics.get)
    best_k_aic = min(aics, key=aics.get)

    result = {
        "bic_1": bics[1],
        "bic_2": bics[2],
        "bic_3": bics[3],
        "delta_bic_1_minus_2": bics[1] - bics[2],
        "delta_bic_2_minus_3": bics[2] - bics[3],
        "best_k_bic": int(best_k_bic),
        "aic_1": aics[1],
        "aic_2": aics[2],
        "aic_3": aics[3],
        "best_k_aic": int(best_k_aic),
        "two_component_mode_count": int(mode_count),
        "is_bimodal_2component": bool(mode_count >= 2),
        "boundary": float(boundary) if boundary is not None else np.nan,
        **p2,
    }

    if boundary is None:
        result["high_count_at_boundary"] = np.nan
        result["high_rate_at_boundary"] = np.nan
        result["boundary_percentile"] = np.nan
    else:
        high = x >= boundary
        result["high_count_at_boundary"] = int(high.sum())
        result["high_rate_at_boundary"] = float(high.mean())
        result["boundary_percentile"] = float((x < boundary).mean())

    return result, models


def cluster_bootstrap_frame(
    train: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    if "player_key" not in train.columns:
        idx = rng.integers(0, len(train), size=len(train))
        return train.iloc[idx].copy()

    grouped = {key: g.copy() for key, g in train.groupby("player_key", sort=False)}
    keys = list(grouped)
    sampled_keys = rng.choice(keys, size=len(keys), replace=True)

    # Preserve every contract belonging to each sampled player cluster.
    frames = [grouped[key] for key in sampled_keys]
    return pd.concat(frames, ignore_index=True)


def bootstrap_role(train: pd.DataFrame, role: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for b in range(N_BOOTSTRAP):
        sample = cluster_bootstrap_frame(train, rng)
        x = sample["composite_score"].to_numpy(dtype=float)

        try:
            result, _ = evaluate_distribution(x, seed=RANDOM_SEED + b + 1)
            rows.append(
                {
                    "role": role,
                    "bootstrap": b,
                    "n": len(sample),
                    "delta_bic_1_minus_2": result["delta_bic_1_minus_2"],
                    "delta_bic_2_minus_3": result["delta_bic_2_minus_3"],
                    "best_k_bic": result["best_k_bic"],
                    "ashman_d": result["ashman_d"],
                    "is_bimodal_2component": result["is_bimodal_2component"],
                    "boundary": result["boundary"],
                    "high_rate_at_boundary": result["high_rate_at_boundary"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "role": role,
                    "bootstrap": b,
                    "n": len(sample),
                    "error": repr(exc),
                }
            )

    return pd.DataFrame(rows)


def summarize_bootstrap(draws: pd.DataFrame) -> dict:
    valid = draws.loc[draws["error"].isna()] if "error" in draws else draws.copy()
    boundary_valid = valid.loc[np.isfinite(valid["boundary"])].copy()

    out = {
        "bootstrap_total": int(len(draws)),
        "bootstrap_valid": int(len(valid)),
        "bootstrap_failure_rate": float(1 - len(valid) / len(draws))
        if len(draws)
        else np.nan,
        "p_bic2_better_than_1": float(
            (valid["delta_bic_1_minus_2"] > 0).mean()
        )
        if len(valid)
        else np.nan,
        "p_strong_bic2_over_1_delta_ge_10": float(
            (valid["delta_bic_1_minus_2"] >= 10).mean()
        )
        if len(valid)
        else np.nan,
        "p_bic_best_k_2": float((valid["best_k_bic"] == 2).mean())
        if len(valid)
        else np.nan,
        "p_two_component_density_bimodal": float(
            valid["is_bimodal_2component"].mean()
        )
        if len(valid)
        else np.nan,
        "ashman_d_median": float(valid["ashman_d"].median())
        if len(valid)
        else np.nan,
        "ashman_d_ci_low": float(valid["ashman_d"].quantile(0.025))
        if len(valid)
        else np.nan,
        "ashman_d_ci_high": float(valid["ashman_d"].quantile(0.975))
        if len(valid)
        else np.nan,
        "boundary_valid_rate": float(len(boundary_valid) / len(valid))
        if len(valid)
        else np.nan,
        "boundary_median": float(boundary_valid["boundary"].median())
        if len(boundary_valid)
        else np.nan,
        "boundary_ci_low": float(boundary_valid["boundary"].quantile(0.025))
        if len(boundary_valid)
        else np.nan,
        "boundary_ci_high": float(boundary_valid["boundary"].quantile(0.975))
        if len(boundary_valid)
        else np.nan,
        "high_rate_median": float(boundary_valid["high_rate_at_boundary"].median())
        if len(boundary_valid)
        else np.nan,
        "high_rate_ci_low": float(
            boundary_valid["high_rate_at_boundary"].quantile(0.025)
        )
        if len(boundary_valid)
        else np.nan,
        "high_rate_ci_high": float(
            boundary_valid["high_rate_at_boundary"].quantile(0.975)
        )
        if len(boundary_valid)
        else np.nan,
    }
    return out


def make_plot(
    role_label: str,
    x: np.ndarray,
    models: dict[int, GaussianMixture],
    result: dict,
    output_path: Path,
) -> None:
    _, grid, _ = count_visible_modes(models[2], x)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.hist(x, bins="auto", density=True, alpha=0.35, label="Training Composite")

    for k in (1, 2, 3):
        ax.plot(grid, mixture_pdf(models[k], grid), label=f"{k}-component GMM")

    # Show individual 2-component densities.
    p = sorted_two_component_params(models[2])
    low_pdf = p["low_weight"] * norm.pdf(
        grid, loc=p["low_mean"], scale=p["low_sd"]
    )
    high_pdf = p["high_weight"] * norm.pdf(
        grid, loc=p["high_mean"], scale=p["high_sd"]
    )
    ax.plot(grid, low_pdf, linestyle="--", label="2-GMM low component")
    ax.plot(grid, high_pdf, linestyle="--", label="2-GMM high component")

    if np.isfinite(result["boundary"]):
        ax.axvline(
            result["boundary"],
            linestyle=":",
            linewidth=1.6,
            label=(
                f"Posterior boundary={result['boundary']:.3f} "
                f"(High={result['high_rate_at_boundary']:.1%})"
            ),
        )

    ax.set_title(
        f"{role_label}: training Composite distribution\n"
        f"ΔBIC(1−2)={result['delta_bic_1_minus_2']:.1f}, "
        f"best BIC k={result['best_k_bic']}, "
        f"Ashman D={result['ashman_d']:.2f}, "
        f"2-GMM modes={result['two_component_mode_count']}"
    )
    ax.set_xlabel("Composite Score (0.5 zWAR + 0.5 zUtilization)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_report(summary: pd.DataFrame, boot: pd.DataFrame) -> str:
    lines = [
        "# Composite Score distribution-structure check",
        "",
        "This analysis uses only 2012–2017 training contracts. "
        "It does not use validation/test predictive performance to define the outcome.",
        "",
        "## Conservative interpretation rule",
        "",
        "The script marks `clean_2group_support=True` only when all of the following hold:",
        "",
        "- ΔBIC(1−2) ≥ 10",
        "- k=2 has the best BIC among k=1,2,3",
        "- the fitted 2-component density has two visible modes",
        "- Ashman's D ≥ 2",
        "- an equal-posterior boundary exists between the two means",
        "- bootstrap P(BIC2 < BIC1) ≥ 0.80",
        "- bootstrap P(two visible modes) ≥ 0.70",
        "",
        "These are conservative heuristics for construct defensibility, not a universal formal test.",
        "",
        "## Results",
        "",
        "| Role | ΔBIC 1−2 | Best k (BIC) | Modes in 2-GMM | Ashman D | Boundary | High rate | Bootstrap P(BIC2<1) | Bootstrap P(bimodal) | Clean 2-group support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    merged = summary.merge(boot, on="role", how="left")
    for _, r in merged.iterrows():
        boundary = "NA" if not np.isfinite(r["boundary"]) else f"{r['boundary']:.3f}"
        high_rate = (
            "NA"
            if not np.isfinite(r["high_rate_at_boundary"])
            else f"{r['high_rate_at_boundary']:.1%}"
        )
        lines.append(
            f"| {r['role_label']} | {r['delta_bic_1_minus_2']:.1f} | "
            f"{int(r['best_k_bic'])} | {int(r['two_component_mode_count'])} | "
            f"{r['ashman_d']:.2f} | {boundary} | {high_rate} | "
            f"{r['p_bic2_better_than_1']:.1%} | "
            f"{r['p_two_component_density_bimodal']:.1%} | "
            f"{bool(r['clean_2group_support'])} |"
        )

    lines += [
        "",
        "## How to interpret",
        "",
        "- If `clean_2group_support=True`, a distribution-derived binary boundary may be defensible, "
        "but it should still be described as a training-distribution boundary rather than a universal baseball success threshold.",
        "- If BIC favors 2 components but the fitted density has only one visible mode, the second component may simply be modeling skewness/tails. "
        "Do not call that a natural two-group split.",
        "- If k=3 has lower BIC than k=2, the data do not support a uniquely preferred binary structure.",
        "- If the bootstrap boundary or High proportion is unstable, avoid using the single-sample intersection as the main paper cutoff.",
        "- If most roles fail these checks, a percentile-based relative-performance definition or a continuous outcome is more defensible than claiming a natural binary cutoff.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    bootstrap_frames = []
    bootstrap_summary_rows = []

    for role, cfg in ROLE_CONFIG.items():
        print("\n" + "=" * 78)
        print(cfg["label"])
        print("=" * 78)

        train, base_params = load_training_composite(role, cfg)
        x = train["composite_score"].to_numpy(dtype=float)

        result, fitted_models = evaluate_distribution(x)
        summary_row = {
            "role": role,
            "role_label": cfg["label"],
            **base_params,
            **result,
        }
        summary_rows.append(summary_row)

        print(
            f"n={len(x)} | ΔBIC(1-2)={result['delta_bic_1_minus_2']:.3f} | "
            f"best k={result['best_k_bic']} | "
            f"2-GMM modes={result['two_component_mode_count']} | "
            f"Ashman D={result['ashman_d']:.3f}"
        )
        if np.isfinite(result["boundary"]):
            print(
                f"Boundary={result['boundary']:.4f} | "
                f"High rate={result['high_rate_at_boundary']:.3%}"
            )
        else:
            print("No equal-posterior boundary between the component means.")

        make_plot(
            cfg["label"],
            x,
            fitted_models,
            result,
            OUT_DIR / f"{role}_gmm_structure.png",
        )

        print(f"Running {N_BOOTSTRAP} player-cluster bootstrap repetitions...")
        draws = bootstrap_role(train, role)
        bootstrap_frames.append(draws)

        boot_summary = {"role": role, **summarize_bootstrap(draws)}
        bootstrap_summary_rows.append(boot_summary)

    summary = pd.DataFrame(summary_rows)
    bootstrap_draws = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_summary = pd.DataFrame(bootstrap_summary_rows)

    combined = summary.merge(bootstrap_summary, on="role", how="left")

    combined["clean_2group_support"] = (
        (combined["delta_bic_1_minus_2"] >= 10)
        & (combined["best_k_bic"] == 2)
        & (combined["is_bimodal_2component"])
        & (combined["ashman_d"] >= 2)
        & (combined["boundary"].notna())
        & (combined["p_bic2_better_than_1"] >= 0.80)
        & (combined["p_two_component_density_bimodal"] >= 0.70)
    )

    summary.to_csv(
        OUT_DIR / "gmm_single_fit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_draws.to_csv(
        OUT_DIR / "gmm_bootstrap_draws.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_summary.to_csv(
        OUT_DIR / "gmm_bootstrap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combined.to_csv(
        OUT_DIR / "gmm_combined_decision_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = make_report(summary, bootstrap_summary)
    # Recompute the report table with decision flag included.
    merged_for_report = summary.merge(bootstrap_summary, on="role", how="left")
    merged_for_report["clean_2group_support"] = combined["clean_2group_support"].to_numpy()
    report = make_report(summary, bootstrap_summary).replace(
        "## How to interpret",
        "## Decision flags\n\n"
        + "\n".join(
            f"- {row.role_label}: clean_2group_support = "
            f"{bool(row.clean_2group_support)}"
            for row in merged_for_report.itertuples()
        )
        + "\n\n## How to interpret"
    )
    (OUT_DIR / "COMPOSITE_STRUCTURE_REPORT.md").write_text(
        report, encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    cols = [
        "role_label",
        "n_contracts",
        "delta_bic_1_minus_2",
        "best_k_bic",
        "two_component_mode_count",
        "ashman_d",
        "boundary",
        "high_rate_at_boundary",
        "p_bic2_better_than_1",
        "p_two_component_density_bimodal",
        "clean_2group_support",
    ]
    print(combined[cols].to_string(index=False))

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
