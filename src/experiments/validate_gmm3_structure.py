# -*- coding: utf-8 -*-
"""
Validate whether the 2012-2017 training Composite Score distribution supports
a stable THREE-group latent structure (Low / Middle / High).

This is a diagnostic/construct-validation experiment.
It does NOT use Validation/Test predictive performance to choose outcome labels.

Main safeguards
---------------
1) Uses only FA years 2012-2017 for distribution fitting.
2) Composite standardization is estimated on that same training period only.
3) Compares GMM k=1..5 by BIC/AIC, so k=3 is not favored by construction.
4) For the 3-component candidate, checks:
   - whether k=3 is actually BIC-best
   - BIC margin over the second-best k
   - whether the fitted mixture density has 3 visible modes
   - adjacent-component Ashman's D separation
   - component weights / hard-assignment counts
   - posterior assignment confidence / entropy
   - ordered MAP boundaries between Low/Middle/High
   - player-cluster bootstrap stability
   - descriptive WAR / Utilization profiles
   - agreement with the paper's current tertile labels (diagnostic only)
5) Saves all outputs without touching the paper's main modeling outputs.

Interpretation
--------------
Even if this script finds a stable 3-component solution, do NOT call it a
"universal optimal success cutoff." The strongest wording is:
"training-distribution-derived latent performance groups."

If the three-group structure is not stable, do not use GMM boundaries as the
paper's main outcome definition.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed_v2"
OUT_DIR = ROOT / "outputs" / "gmm3_structure_validation"

RANDOM_SEED = 42
N_BOOTSTRAP = 300
GMM_N_INIT = 20
GMM_MAX_ITER = 1500
REG_COVAR = 1e-6
K_CANDIDATES = (1, 2, 3, 4, 5)

# Conservative decision-aid thresholds.
# These are heuristics for "strong evidence", not universal statistical laws.
MIN_BIC_MARGIN = 10.0
MIN_COMPONENT_WEIGHT = 0.10
MIN_HARD_COUNT = 10
MIN_ADJACENT_ASHMAN_D = 2.0
MIN_MEAN_MAX_POSTERIOR = 0.70
MIN_BOOT_P_K3_BEST = 0.70
MIN_BOOT_P_THREE_MODES = 0.70
MIN_BOOT_BOUNDARY_VALID = 0.90

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


def fit_training_composite(role: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    path = DATA_DIR / cfg["file"]
    if not path.exists():
        raise FileNotFoundError(f"Missing cohort file: {path}")

    df = pd.read_csv(path)
    required = {"fa_year", cfg["war"], cfg["utilization"]}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path.name}: missing columns {sorted(missing)}")

    df["fa_year"] = pd.to_numeric(df["fa_year"], errors="coerce")
    train = df.loc[df["fa_year"].between(2012, 2017)].copy()
    if train.empty:
        raise ValueError(f"{role}: no training observations in 2012-2017")

    params = {}
    for col, short in ((cfg["war"], "war"), (cfg["utilization"], "utilization")):
        values = pd.to_numeric(train[col], errors="coerce")
        median = float(values.median())
        filled = values.fillna(median)
        mean = float(filled.mean())
        sd = float(filled.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"{role}: invalid SD for {col}: {sd}")
        train[f"z_{short}"] = (filled - mean) / sd
        params.update({
            f"{short}_median": median,
            f"{short}_mean": mean,
            f"{short}_sd": sd,
        })

    train["composite_score"] = (
        0.5 * train["z_war"] + 0.5 * train["z_utilization"]
    )
    params.update({
        "n_contracts": int(len(train)),
        "n_players": int(train["player_key"].nunique())
        if "player_key" in train.columns else int(len(train)),
        "min_year": int(train["fa_year"].min()),
        "max_year": int(train["fa_year"].max()),
        "composite_mean": float(train["composite_score"].mean()),
        "composite_sd": float(train["composite_score"].std(ddof=0)),
        "composite_skew": float(train["composite_score"].skew()),
    })
    return train, params


def fit_gmm(x: np.ndarray, k: int, seed: int) -> GaussianMixture:
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


def component_arrays(model: GaussianMixture):
    means = model.means_.reshape(-1)
    variances = model.covariances_.reshape(-1)
    weights = model.weights_.reshape(-1)
    order = np.argsort(means)
    return (
        means[order],
        np.sqrt(variances[order]),
        weights[order],
        order,
    )


def ashman_d(m1: float, s1: float, m2: float, s2: float) -> float:
    return (
        math.sqrt(2.0)
        * abs(m2 - m1)
        / math.sqrt(s1**2 + s2**2)
    )


def ordered_posteriors(model: GaussianMixture, x: np.ndarray):
    _, _, _, order = component_arrays(model)
    raw = model.predict_proba(x.reshape(-1, 1))
    return raw[:, order]


def ordered_map_labels(model: GaussianMixture, x: np.ndarray):
    p = ordered_posteriors(model, x)
    return np.argmax(p, axis=1), p


def mixture_pdf(model: GaussianMixture, grid: np.ndarray) -> np.ndarray:
    return np.exp(model.score_samples(grid.reshape(-1, 1)))


def count_visible_modes(model: GaussianMixture, x: np.ndarray):
    sd = float(np.std(x, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    lo = float(np.min(x) - 1.5 * sd)
    hi = float(np.max(x) + 1.5 * sd)
    grid = np.linspace(lo, hi, 6000)
    density = mixture_pdf(model, grid)
    prominence = max(float(density.max()) * 0.02, 1e-12)
    peaks, _ = find_peaks(density, prominence=prominence)
    return int(len(peaks)), grid, density, peaks


def map_boundaries(model: GaussianMixture, x: np.ndarray):
    means, sds, weights, _ = component_arrays(model)
    sd = float(np.std(x, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    lo = min(float(np.min(x) - 2 * sd), float(means[0] - 4 * sds[0]))
    hi = max(float(np.max(x) + 2 * sd), float(means[-1] + 4 * sds[-1]))
    grid = np.linspace(lo, hi, 20000)

    p = ordered_posteriors(model, grid)
    labels = np.argmax(p, axis=1)

    # Find ordered label transitions 0->1 and 1->2.
    boundaries = []
    for left_label, right_label in ((0, 1), (1, 2)):
        idx = np.flatnonzero(
            (labels[:-1] == left_label) & (labels[1:] == right_label)
        )
        if len(idx) == 0:
            boundaries.append(np.nan)
        else:
            # Prefer transition nearest midpoint of adjacent component means.
            midpoint = (means[left_label] + means[right_label]) / 2
            candidates = [(grid[i] + grid[i + 1]) / 2 for i in idx]
            boundaries.append(
                float(min(candidates, key=lambda z: abs(z - midpoint)))
            )
    return boundaries[0], boundaries[1]


def normalized_entropy(posteriors: np.ndarray) -> np.ndarray:
    eps = 1e-15
    p = np.clip(posteriors, eps, 1.0)
    ent = -np.sum(p * np.log(p), axis=1)
    return ent / np.log(p.shape[1])


def tertile_labels(x: np.ndarray):
    q1, q2 = np.quantile(x, [1/3, 2/3])
    return np.select([x < q1, x < q2], [0, 1], default=2).astype(int), q1, q2


def evaluate_one_sample(x: np.ndarray, seed: int):
    models = {k: fit_gmm(x, k, seed + 100 * k) for k in K_CANDIDATES}
    bics = {k: float(models[k].bic(x.reshape(-1, 1))) for k in K_CANDIDATES}
    aics = {k: float(models[k].aic(x.reshape(-1, 1))) for k in K_CANDIDATES}
    best_k = min(bics, key=bics.get)
    ordered_bic = sorted(bics.items(), key=lambda kv: kv[1])
    bic_margin = ordered_bic[1][1] - ordered_bic[0][1]

    g3 = models[3]
    means, sds, weights, _ = component_arrays(g3)
    labels, post = ordered_map_labels(g3, x)
    mode_count, grid, density, peaks = count_visible_modes(g3, x)
    low_mid_d = ashman_d(means[0], sds[0], means[1], sds[1])
    mid_high_d = ashman_d(means[1], sds[1], means[2], sds[2])
    b1, b2 = map_boundaries(g3, x)

    counts = np.bincount(labels, minlength=3)
    max_post = post.max(axis=1)
    ent = normalized_entropy(post)
    tertile, q1, q2 = tertile_labels(x)

    result = {
        **{f"bic_{k}": bics[k] for k in K_CANDIDATES},
        **{f"aic_{k}": aics[k] for k in K_CANDIDATES},
        "best_k_bic": int(best_k),
        "bic_margin_best_vs_second": float(bic_margin),
        "gmm3_mode_count": int(mode_count),
        "low_mean": float(means[0]),
        "middle_mean": float(means[1]),
        "high_mean": float(means[2]),
        "low_sd": float(sds[0]),
        "middle_sd": float(sds[1]),
        "high_sd": float(sds[2]),
        "low_weight": float(weights[0]),
        "middle_weight": float(weights[1]),
        "high_weight": float(weights[2]),
        "low_mid_ashman_d": float(low_mid_d),
        "mid_high_ashman_d": float(mid_high_d),
        "boundary_low_middle": float(b1) if np.isfinite(b1) else np.nan,
        "boundary_middle_high": float(b2) if np.isfinite(b2) else np.nan,
        "low_count": int(counts[0]),
        "middle_count": int(counts[1]),
        "high_count": int(counts[2]),
        "high_rate": float((labels == 2).mean()),
        "mean_max_posterior": float(max_post.mean()),
        "median_max_posterior": float(np.median(max_post)),
        "p_max_posterior_ge_0_70": float((max_post >= 0.70).mean()),
        "mean_normalized_entropy": float(ent.mean()),
        "tertile_q1": float(q1),
        "tertile_q2": float(q2),
        "tertile_gmm_agreement": float((tertile == labels).mean()),
        "tertile_gmm_ari": float(adjusted_rand_score(tertile, labels)),
    }
    return result, models, labels, post, grid, density, peaks


def add_profiles(
    train: pd.DataFrame, labels: np.ndarray, cfg: dict, role: str
):
    tmp = train.copy()
    tmp["gmm3_class"] = labels
    names = {0: "Low", 1: "Middle", 2: "High"}
    rows = []

    for cls in (0, 1, 2):
        g = tmp.loc[tmp["gmm3_class"].eq(cls)]
        rows.append({
            "role": role,
            "class_id": cls,
            "class_name": names[cls],
            "n": len(g),
            "share": len(g) / len(tmp),
            "composite_median": float(g["composite_score"].median()),
            "composite_q25": float(g["composite_score"].quantile(.25)),
            "composite_q75": float(g["composite_score"].quantile(.75)),
            "war_median": float(pd.to_numeric(g[cfg["war"]], errors="coerce").median()),
            "war_q25": float(pd.to_numeric(g[cfg["war"]], errors="coerce").quantile(.25)),
            "war_q75": float(pd.to_numeric(g[cfg["war"]], errors="coerce").quantile(.75)),
            "utilization_median": float(pd.to_numeric(g[cfg["utilization"]], errors="coerce").median()),
            "utilization_q25": float(pd.to_numeric(g[cfg["utilization"]], errors="coerce").quantile(.25)),
            "utilization_q75": float(pd.to_numeric(g[cfg["utilization"]], errors="coerce").quantile(.75)),
        })
    return pd.DataFrame(rows), tmp


def cluster_bootstrap_frame(train: pd.DataFrame, rng: np.random.Generator):
    if "player_key" not in train.columns:
        idx = rng.integers(0, len(train), size=len(train))
        return train.iloc[idx].copy()

    groups = {key: g.copy() for key, g in train.groupby("player_key", sort=False)}
    keys = np.array(list(groups), dtype=object)
    sampled = rng.choice(keys, size=len(keys), replace=True)
    return pd.concat([groups[key] for key in sampled], ignore_index=True)


def bootstrap_role(train: pd.DataFrame, role: str):
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for b in range(N_BOOTSTRAP):
        sample = cluster_bootstrap_frame(train, rng)
        x = sample["composite_score"].to_numpy(dtype=float)
        row = {"role": role, "bootstrap": b, "n": len(sample)}
        try:
            result, _, _, _, _, _, _ = evaluate_one_sample(
                x, RANDOM_SEED + 10000 + b
            )
            row.update({
                "best_k_bic": result["best_k_bic"],
                "bic_margin_best_vs_second": result["bic_margin_best_vs_second"],
                "gmm3_mode_count": result["gmm3_mode_count"],
                "low_mid_ashman_d": result["low_mid_ashman_d"],
                "mid_high_ashman_d": result["mid_high_ashman_d"],
                "low_weight": result["low_weight"],
                "middle_weight": result["middle_weight"],
                "high_weight": result["high_weight"],
                "boundary_low_middle": result["boundary_low_middle"],
                "boundary_middle_high": result["boundary_middle_high"],
                "high_rate": result["high_rate"],
                "mean_max_posterior": result["mean_max_posterior"],
                "error": None,
            })
        except Exception as exc:
            row["error"] = repr(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_summary(draws: pd.DataFrame):
    valid = draws.loc[draws["error"].isna()].copy()
    if valid.empty:
        return {
            "bootstrap_total": len(draws),
            "bootstrap_valid": 0,
            "bootstrap_failure_rate": 1.0,
        }

    boundaries_ok = (
        np.isfinite(valid["boundary_low_middle"])
        & np.isfinite(valid["boundary_middle_high"])
        & (valid["boundary_low_middle"] < valid["boundary_middle_high"])
    )

    out = {
        "bootstrap_total": int(len(draws)),
        "bootstrap_valid": int(len(valid)),
        "bootstrap_failure_rate": float(1 - len(valid)/len(draws)),
        "p_best_k_3": float((valid["best_k_bic"] == 3).mean()),
        "p_best_k_3_margin_ge_10": float(
            ((valid["best_k_bic"] == 3) &
             (valid["bic_margin_best_vs_second"] >= MIN_BIC_MARGIN)).mean()
        ),
        "p_three_visible_modes": float((valid["gmm3_mode_count"] >= 3).mean()),
        "p_both_adjacent_D_ge_2": float(
            ((valid["low_mid_ashman_d"] >= MIN_ADJACENT_ASHMAN_D) &
             (valid["mid_high_ashman_d"] >= MIN_ADJACENT_ASHMAN_D)).mean()
        ),
        "p_all_weights_ge_0_10": float(
            ((valid[["low_weight","middle_weight","high_weight"]]
              .min(axis=1)) >= MIN_COMPONENT_WEIGHT).mean()
        ),
        "boundary_valid_rate": float(boundaries_ok.mean()),
        "boundary_low_middle_median": float(
            valid.loc[boundaries_ok, "boundary_low_middle"].median()
        ) if boundaries_ok.any() else np.nan,
        "boundary_low_middle_ci_low": float(
            valid.loc[boundaries_ok, "boundary_low_middle"].quantile(.025)
        ) if boundaries_ok.any() else np.nan,
        "boundary_low_middle_ci_high": float(
            valid.loc[boundaries_ok, "boundary_low_middle"].quantile(.975)
        ) if boundaries_ok.any() else np.nan,
        "boundary_middle_high_median": float(
            valid.loc[boundaries_ok, "boundary_middle_high"].median()
        ) if boundaries_ok.any() else np.nan,
        "boundary_middle_high_ci_low": float(
            valid.loc[boundaries_ok, "boundary_middle_high"].quantile(.025)
        ) if boundaries_ok.any() else np.nan,
        "boundary_middle_high_ci_high": float(
            valid.loc[boundaries_ok, "boundary_middle_high"].quantile(.975)
        ) if boundaries_ok.any() else np.nan,
        "high_rate_median": float(valid["high_rate"].median()),
        "high_rate_ci_low": float(valid["high_rate"].quantile(.025)),
        "high_rate_ci_high": float(valid["high_rate"].quantile(.975)),
        "mean_max_posterior_median": float(valid["mean_max_posterior"].median()),
    }
    return out


def plot_role(
    role_label: str,
    x: np.ndarray,
    model3: GaussianMixture,
    result: dict,
    grid: np.ndarray,
    density: np.ndarray,
    peaks: np.ndarray,
    output: Path,
):
    means, sds, weights, _ = component_arrays(model3)

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.hist(x, bins="auto", density=True, alpha=0.35, label="Training Composite")
    ax.plot(grid, density, linewidth=2, label="3-component GMM mixture")

    names = ["Low component", "Middle component", "High component"]
    for i in range(3):
        comp = weights[i] * norm.pdf(grid, loc=means[i], scale=sds[i])
        ax.plot(grid, comp, linestyle="--", label=names[i])

    for boundary, name in (
        (result["boundary_low_middle"], "Low/Middle MAP boundary"),
        (result["boundary_middle_high"], "Middle/High MAP boundary"),
    ):
        if np.isfinite(boundary):
            ax.axvline(boundary, linestyle=":", linewidth=1.5, label=name)

    if len(peaks):
        ax.scatter(grid[peaks], density[peaks], marker="o", zorder=5, label="Visible modes")

    ax.set_title(
        f"{role_label}: 3-component structure check\n"
        f"best BIC k={result['best_k_bic']}, "
        f"BIC margin={result['bic_margin_best_vs_second']:.1f}, "
        f"modes={result['gmm3_mode_count']}, "
        f"D(L-M)={result['low_mid_ashman_d']:.2f}, "
        f"D(M-H)={result['mid_high_ashman_d']:.2f}"
    )
    ax.set_xlabel("Composite Score (0.5 zWAR + 0.5 zUtilization)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    single_rows = []
    profile_frames = []
    assignment_frames = []
    bootstrap_frames = []
    bootstrap_summary_rows = []

    for role, cfg in ROLE_CONFIG.items():
        print("\n" + "=" * 84)
        print(cfg["label"])
        print("=" * 84)

        train, base = fit_training_composite(role, cfg)
        x = train["composite_score"].to_numpy(dtype=float)

        result, models, labels, post, grid, density, peaks = evaluate_one_sample(
            x, RANDOM_SEED
        )
        single_rows.append({
            "role": role,
            "role_label": cfg["label"],
            **base,
            **result,
        })

        profiles, assigned = add_profiles(train, labels, cfg, role)
        profile_frames.append(profiles)

        ids = [c for c in ["player_key", "player", "fa_year"] if c in assigned.columns]
        a = assigned[ids + ["composite_score"]].copy()
        a["role"] = role
        a["gmm3_class"] = labels
        a["gmm3_class_name"] = pd.Series(labels).map({0:"Low",1:"Middle",2:"High"}).to_numpy()
        a["posterior_low"] = post[:, 0]
        a["posterior_middle"] = post[:, 1]
        a["posterior_high"] = post[:, 2]
        assignment_frames.append(a)

        print(
            f"n={len(x)} | best BIC k={result['best_k_bic']} | "
            f"BIC margin={result['bic_margin_best_vs_second']:.3f} | "
            f"3-GMM modes={result['gmm3_mode_count']}"
        )
        print(
            f"D Low-Middle={result['low_mid_ashman_d']:.3f} | "
            f"D Middle-High={result['mid_high_ashman_d']:.3f}"
        )
        print(
            f"Weights={result['low_weight']:.3f} / "
            f"{result['middle_weight']:.3f} / "
            f"{result['high_weight']:.3f}"
        )
        print(
            f"Boundaries={result['boundary_low_middle']:.4f}, "
            f"{result['boundary_middle_high']:.4f} | "
            f"High rate={result['high_rate']:.3%} | "
            f"Mean max posterior={result['mean_max_posterior']:.3f}"
        )
        print(
            f"Tertile agreement={result['tertile_gmm_agreement']:.3f} | "
            f"ARI={result['tertile_gmm_ari']:.3f}"
        )

        plot_role(
            cfg["label"], x, models[3], result, grid, density, peaks,
            OUT_DIR / f"{role}_gmm3_structure.png"
        )

        print(f"Running {N_BOOTSTRAP} player-cluster bootstrap repetitions...")
        draws = bootstrap_role(train, role)
        bootstrap_frames.append(draws)
        bootstrap_summary_rows.append({
            "role": role,
            **bootstrap_summary(draws),
        })

    single = pd.DataFrame(single_rows)
    profiles = pd.concat(profile_frames, ignore_index=True)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    boot_draws = pd.concat(bootstrap_frames, ignore_index=True)
    boot_summary = pd.DataFrame(bootstrap_summary_rows)
    combined = single.merge(boot_summary, on="role", how="left")

    combined["all_component_weights_ge_min"] = (
        combined[["low_weight","middle_weight","high_weight"]].min(axis=1)
        >= MIN_COMPONENT_WEIGHT
    )
    combined["all_hard_counts_ge_min"] = (
        combined[["low_count","middle_count","high_count"]].min(axis=1)
        >= MIN_HARD_COUNT
    )
    combined["both_adjacent_D_ge_2"] = (
        (combined["low_mid_ashman_d"] >= MIN_ADJACENT_ASHMAN_D)
        & (combined["mid_high_ashman_d"] >= MIN_ADJACENT_ASHMAN_D)
    )
    combined["valid_ordered_boundaries"] = (
        np.isfinite(combined["boundary_low_middle"])
        & np.isfinite(combined["boundary_middle_high"])
        & (combined["boundary_low_middle"] < combined["boundary_middle_high"])
    )

    # Strong, conservative support flag.
    combined["clean_3group_support"] = (
        (combined["best_k_bic"] == 3)
        & (combined["bic_margin_best_vs_second"] >= MIN_BIC_MARGIN)
        & (combined["gmm3_mode_count"] >= 3)
        & combined["both_adjacent_D_ge_2"]
        & combined["all_component_weights_ge_min"]
        & combined["all_hard_counts_ge_min"]
        & (combined["mean_max_posterior"] >= MIN_MEAN_MAX_POSTERIOR)
        & combined["valid_ordered_boundaries"]
        & (combined["p_best_k_3"] >= MIN_BOOT_P_K3_BEST)
        & (combined["p_three_visible_modes"] >= MIN_BOOT_P_THREE_MODES)
        & (combined["boundary_valid_rate"] >= MIN_BOOT_BOUNDARY_VALID)
    )

    single.to_csv(
        OUT_DIR / "gmm3_single_fit_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    profiles.to_csv(
        OUT_DIR / "gmm3_component_profiles.csv",
        index=False, encoding="utf-8-sig"
    )
    assignments.to_csv(
        OUT_DIR / "gmm3_training_assignments.csv",
        index=False, encoding="utf-8-sig"
    )
    boot_draws.to_csv(
        OUT_DIR / "gmm3_bootstrap_draws.csv",
        index=False, encoding="utf-8-sig"
    )
    boot_summary.to_csv(
        OUT_DIR / "gmm3_bootstrap_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    combined.to_csv(
        OUT_DIR / "gmm3_combined_decision_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    report = []
    report += [
        "# Three-component Composite structure validation",
        "",
        "Only 2012–2017 training contracts were used for mixture fitting.",
        "Validation/Test prediction performance was not used to choose the outcome groups.",
        "",
        "## Main decision table",
        "",
        "| Role | Best k | BIC margin | Modes | D L-M | D M-H | Weights L/M/H | High rate | Mean max posterior | Boot P(k=3 best) | Boot P(3 modes) | Clean 3-group support |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in combined.itertuples():
        report.append(
            f"| {r.role_label} | {int(r.best_k_bic)} | "
            f"{r.bic_margin_best_vs_second:.1f} | {int(r.gmm3_mode_count)} | "
            f"{r.low_mid_ashman_d:.2f} | {r.mid_high_ashman_d:.2f} | "
            f"{r.low_weight:.2f}/{r.middle_weight:.2f}/{r.high_weight:.2f} | "
            f"{r.high_rate:.1%} | {r.mean_max_posterior:.2f} | "
            f"{r.p_best_k_3:.1%} | {r.p_three_visible_modes:.1%} | "
            f"{bool(r.clean_3group_support)} |"
        )
    report += [
        "",
        "## Interpretation rules",
        "",
        "- `best_k_bic=3` alone is not enough.",
        "- A genuine three-group interpretation is strongest when the 3-GMM density has three visible modes, adjacent components are well separated, assignments are confident, and the same structure reappears under player-cluster bootstrap.",
        "- Component profiles should also be inspected. If Low/Middle/High do not show sensible WAR/Utilization patterns, do not describe them as performance levels merely because the Composite means are ordered.",
        "- If `clean_3group_support=False` for a role, avoid using GMM boundaries as the main paper label for that role.",
        "",
        "## Files to inspect",
        "",
        "- `gmm3_combined_decision_summary.csv`",
        "- `gmm3_component_profiles.csv`",
        "- `gmm3_bootstrap_summary.csv`",
        "- `{role}_gmm3_structure.png`",
        "",
    ]
    (OUT_DIR / "GMM3_STRUCTURE_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    cols = [
        "role_label",
        "best_k_bic",
        "bic_margin_best_vs_second",
        "gmm3_mode_count",
        "low_mid_ashman_d",
        "mid_high_ashman_d",
        "low_weight",
        "middle_weight",
        "high_weight",
        "boundary_low_middle",
        "boundary_middle_high",
        "high_rate",
        "mean_max_posterior",
        "p_best_k_3",
        "p_three_visible_modes",
        "p_both_adjacent_D_ge_2",
        "boundary_valid_rate",
        "clean_3group_support",
    ]

    print("\n" + "=" * 84)
    print("FINAL 3-COMPONENT SUMMARY")
    print("=" * 84)
    print(combined[cols].to_string(index=False))

    print("\n" + "=" * 84)
    print("COMPONENT PROFILES")
    print("=" * 84)
    print(profiles.to_string(index=False))

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
