"""Paired bootstrap comparison of final Composite vs WAR-only 2-class models.

Reads final test predictions produced after validation-only cutoff/model/threshold
selection. Reports both observation-level and player-cluster paired bootstrap.

Metrics bootstrapped:
- Accuracy difference (Composite - WAR-only)
- High-class F1 difference (Composite - WAR-only)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "optimized_2class"
PRED_PATH = OUT / "final_predictions.csv"
N_BOOT = 10_000
SEED = 42


def row_components(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in ("composite", "war"):
        y = frame[f"y_true_{name}"].to_numpy(dtype=int)
        p = frame[f"y_pred_{name}"].to_numpy(dtype=int)
        out[f"correct_{name}"] = (y == p).astype(float)
        out[f"tp_{name}"] = ((y == 1) & (p == 1)).astype(float)
        out[f"fp_{name}"] = ((y == 0) & (p == 1)).astype(float)
        out[f"fn_{name}"] = ((y == 1) & (p == 0)).astype(float)
    return out


def f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)


def observed_deltas(frame: pd.DataFrame) -> tuple[float, float]:
    comp_acc = np.mean(frame["y_true_composite"].to_numpy() == frame["y_pred_composite"].to_numpy())
    war_acc = np.mean(frame["y_true_war"].to_numpy() == frame["y_pred_war"].to_numpy())
    comp_f1 = f1_score(frame["y_true_composite"], frame["y_pred_composite"], zero_division=0)
    war_f1 = f1_score(frame["y_true_war"], frame["y_pred_war"], zero_division=0)
    return float(comp_acc - war_acc), float(comp_f1 - war_f1)


def summarize(draws: np.ndarray) -> dict[str, float]:
    return {
        "bootstrap_mean": float(np.mean(draws)),
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def vectorized_bootstrap_from_matrix(
    matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap rows of a component matrix and return delta-Accuracy/delta-F1."""
    # columns:
    # 0 correct_c, 1 tp_c, 2 fp_c, 3 fn_c,
    # 4 correct_w, 5 tp_w, 6 fp_w, 7 fn_w, 8 weight/count
    n_units = matrix.shape[0]
    indices = rng.integers(0, n_units, size=(N_BOOT, n_units))
    sampled = matrix[indices].sum(axis=1)
    total_n = sampled[:, 8]
    comp_acc = sampled[:, 0] / total_n
    war_acc = sampled[:, 4] / total_n
    comp_f1 = f1_from_counts(sampled[:, 1], sampled[:, 2], sampled[:, 3])
    war_f1 = f1_from_counts(sampled[:, 5], sampled[:, 6], sampled[:, 7])
    return comp_acc - war_acc, comp_f1 - war_f1


def observation_matrix(frame: pd.DataFrame) -> np.ndarray:
    c = row_components(frame)
    return np.column_stack(
        [
            c["correct_composite"], c["tp_composite"], c["fp_composite"], c["fn_composite"],
            c["correct_war"], c["tp_war"], c["fp_war"], c["fn_war"],
            np.ones(len(frame), dtype=float),
        ]
    )


def player_cluster_matrix(frame: pd.DataFrame) -> np.ndarray:
    temp = frame.copy()
    c = row_components(temp)
    for key, values in c.items():
        temp[key] = values
    temp["n_rows"] = 1.0
    cols = [
        "correct_composite", "tp_composite", "fp_composite", "fn_composite",
        "correct_war", "tp_war", "fp_war", "fn_war", "n_rows",
    ]
    return temp.groupby("player_key", sort=False)[cols].sum().to_numpy(dtype=float)


def paired_role_frame(pred: pd.DataFrame, role: str) -> pd.DataFrame:
    test = pred[(pred["role"].eq(role)) & (pred["split"].eq("test"))].copy()
    comp = test[test["outcome"].eq("composite")].copy()
    war = test[test["outcome"].eq("war")].copy()

    keep = ["contract_row_id", "player_key", "fa_year", "y_true", "y_pred", "probability_high"]
    comp = comp[keep].rename(
        columns={
            "y_true": "y_true_composite",
            "y_pred": "y_pred_composite",
            "probability_high": "probability_composite",
        }
    )
    war = war[keep].rename(
        columns={
            "y_true": "y_true_war",
            "y_pred": "y_pred_war",
            "probability_high": "probability_war",
        }
    )
    merged = comp.merge(
        war,
        on=["contract_row_id", "player_key", "fa_year"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(comp) or len(merged) != len(war):
        raise ValueError(f"{role}: Composite and WAR predictions are not perfectly paired")
    return merged


def main() -> None:
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Run run_optimized_2class.py first: {PRED_PATH}")
    pred = pd.read_csv(PRED_PATH, dtype={"player_key": str})
    rng = np.random.default_rng(SEED)

    rows: list[dict] = []
    paired_frames: list[pd.DataFrame] = []

    for role in ("hitter", "sp", "rp"):
        frame = paired_role_frame(pred, role)
        observed_acc, observed_f1 = observed_deltas(frame)
        paired_frames.append(frame.assign(role=role))

        for method, matrix in (
            ("observation", observation_matrix(frame)),
            ("player_cluster", player_cluster_matrix(frame)),
        ):
            acc_draws, f1_draws = vectorized_bootstrap_from_matrix(matrix, rng)
            rows.append(
                {
                    "role": role,
                    "bootstrap_method": method,
                    "n_contracts": len(frame),
                    "n_players": frame["player_key"].nunique(),
                    "metric": "accuracy",
                    "observed_difference": observed_acc,
                    **summarize(acc_draws),
                }
            )
            rows.append(
                {
                    "role": role,
                    "bootstrap_method": method,
                    "n_contracts": len(frame),
                    "n_players": frame["player_key"].nunique(),
                    "metric": "f1",
                    "observed_difference": observed_f1,
                    **summarize(f1_draws),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(
        OUT / "composite_vs_war_paired_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(paired_frames, ignore_index=True).to_csv(
        OUT / "paired_test_predictions_all_roles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
