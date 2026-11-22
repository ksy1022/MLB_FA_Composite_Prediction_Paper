# -*- coding: utf-8 -*-
"""
Paired bootstrap comparison for the fixed-tertile 2-class experiment.

Run AFTER:
    python src/experiments/run_fixed_tertile_2class.py

Compares selected Composite vs WAR-only TEST predictions within each role.

Reports:
- delta Accuracy = Composite - WAR
- delta F1       = Composite - WAR
using:
1) observation-level paired bootstrap
2) player-cluster paired bootstrap

Note:
Composite and WAR-only have different outcome labels, though both use the same
top-tertile prevalence rule. This analysis compares predictive performance of
the two operationalized outcome definitions; it does not imply that their
individual High labels are identical.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "fixed_tertile_2class"
PRED_PATH = OUT / "selected_predictions.csv"

N_BOOT = 5000
SEED = 42


def metric_pair(frame: pd.DataFrame):
    c = frame[frame["outcome"].eq("composite")]
    w = frame[frame["outcome"].eq("war")]

    keys = [c for c in ["player_key", "fa_year"] if c in frame.columns]
    if not keys:
        raise KeyError("Need player_key/fa_year identifiers for paired bootstrap.")

    c = c[keys + ["y_true", "y_pred"]].rename(
        columns={"y_true": "c_true", "y_pred": "c_pred"}
    )
    w = w[keys + ["y_true", "y_pred"]].rename(
        columns={"y_true": "w_true", "y_pred": "w_pred"}
    )
    paired = c.merge(w, on=keys, how="inner", validate="one_to_one")
    return paired, keys


def deltas(df: pd.DataFrame):
    c_acc = accuracy_score(df["c_true"], df["c_pred"])
    w_acc = accuracy_score(df["w_true"], df["w_pred"])
    c_f1 = f1_score(df["c_true"], df["c_pred"], zero_division=0)
    w_f1 = f1_score(df["w_true"], df["w_pred"], zero_division=0)
    return c_acc - w_acc, c_f1 - w_f1


def summarize(draws: np.ndarray):
    return {
        "bootstrap_mean": float(np.mean(draws)),
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def obs_bootstrap(df: pd.DataFrame, rng: np.random.Generator):
    out_acc = np.empty(N_BOOT)
    out_f1 = np.empty(N_BOOT)
    n = len(df)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx]
        out_acc[b], out_f1[b] = deltas(sample)
    return out_acc, out_f1


def cluster_bootstrap(df: pd.DataFrame, rng: np.random.Generator):
    if "player_key" not in df.columns:
        return obs_bootstrap(df, rng)

    groups = {key: g for key, g in df.groupby("player_key", sort=False)}
    players = np.array(list(groups.keys()), dtype=object)

    out_acc = np.empty(N_BOOT)
    out_f1 = np.empty(N_BOOT)
    for b in range(N_BOOT):
        sampled = rng.choice(players, size=len(players), replace=True)
        sample = pd.concat([groups[p] for p in sampled], ignore_index=True)
        out_acc[b], out_f1[b] = deltas(sample)
    return out_acc, out_f1


def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PRED_PATH}. Run run_fixed_tertile_2class.py first."
        )

    pred = pd.read_csv(PRED_PATH)
    pred = pred[pred["split"].eq("test")].copy()

    rows = []
    draw_frames = []

    for role, role_frame in pred.groupby("role"):
        role_label = role_frame["role_label"].iloc[0]
        paired, keys = metric_pair(role_frame)
        point_acc, point_f1 = deltas(paired)

        rng1 = np.random.default_rng(SEED)
        obs_acc, obs_f1 = obs_bootstrap(paired, rng1)

        rng2 = np.random.default_rng(SEED)
        cluster_acc, cluster_f1 = cluster_bootstrap(paired, rng2)

        for bootstrap_type, arr_acc, arr_f1 in (
            ("observation", obs_acc, obs_f1),
            ("player_cluster", cluster_acc, cluster_f1),
        ):
            for metric, point, draws in (
                ("delta_accuracy", point_acc, arr_acc),
                ("delta_f1", point_f1, arr_f1),
            ):
                rows.append({
                    "role": role,
                    "role_label": role_label,
                    "bootstrap_type": bootstrap_type,
                    "metric": metric,
                    "point_estimate": point,
                    **summarize(draws),
                })

            draw_frames.append(pd.DataFrame({
                "role": role,
                "bootstrap_type": bootstrap_type,
                "iteration": np.arange(N_BOOT),
                "delta_accuracy": arr_acc,
                "delta_f1": arr_f1,
            }))

    result = pd.DataFrame(rows)
    draws = pd.concat(draw_frames, ignore_index=True)

    result.to_csv(
        OUT / "composite_vs_war_paired_bootstrap.csv",
        index=False, encoding="utf-8-sig"
    )
    draws.to_csv(
        OUT / "composite_vs_war_bootstrap_draws.csv",
        index=False, encoding="utf-8-sig"
    )

    print("\nPAIRED BOOTSTRAP: Composite - WAR")
    print(result.to_string(index=False))
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
