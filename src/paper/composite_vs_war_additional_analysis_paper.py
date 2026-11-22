from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed_v2"
OUTPUT_DIR = ROOT / "outputs" / "paper"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 10_000
RANDOM_STATE = 42

ROLE_CONFIG = {
    "hitter": {
        "label": "Hitter",
        "war": "post_WAR_3y_sum",
        "util": "post_PA_3y",
        "util_name": "PA",
        "top": 0.40,
    },
    "sp": {
        "label": "Starting Pitcher",
        "war": "post_WAR_3y_sum",
        "util": "post_IP_3y",
        "util_name": "IP",
        "top": 0.20,
    },
    "rp": {
        "label": "Relief Pitcher",
        "war": "post_WAR_3y_sum",
        "util": "post_IP_3y",
        "util_name": "IP",
        "top": 0.20,
    },
}

# Paper-selected models. Used only if temporal_best_models.csv is unavailable.
PAPER_MODEL_SPEC = {
    ("hitter", "composite", "2class"): "logistic",
    ("sp", "composite", "2class"): "logistic",
    ("rp", "composite", "2class"): "rf",
    ("hitter", "composite", "3class"): "rf",
    ("sp", "composite", "3class"): "rf",
    ("rp", "composite", "3class"): "rf",
    ("hitter", "war", "2class"): "logistic",
    ("sp", "war", "2class"): "logistic",
    ("rp", "war", "2class"): "rf",
    ("hitter", "war", "3class"): "logistic",
    ("sp", "war", "3class"): "rf",
    ("rp", "war", "3class"): "logistic",
}


def temporal_split(df: pd.DataFrame) -> pd.Series:
    split = pd.Series(index=df.index, dtype="object")
    split[df["fa_year"].between(2012, 2017)] = "train"
    split[df["fa_year"].between(2018, 2019)] = "valid"
    split[df["fa_year"].between(2020, 2021)] = "test"
    return split


def player_name_column(df: pd.DataFrame) -> str:
    for col in ("player", "Player", "player_name", "Name", "player_norm", "player_key"):
        if col in df.columns:
            return col
    raise KeyError("Player-name column was not found.")


def prepare_outcomes(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out["split"] = temporal_split(out)
    out = out[out["split"].notna()].copy()

    train_mask = out["split"].eq("train")
    params = {}

    for col, short in ((cfg["war"], "war"), (cfg["util"], "util")):
        median = float(out.loc[train_mask, col].median())
        train_filled = out.loc[train_mask, col].fillna(median)
        mean = float(train_filled.mean())
        std = float(train_filled.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            raise ValueError(f"Invalid training SD for {col}: {std}")

        out[f"z_{short}"] = (out[col].fillna(median) - mean) / std
        params[f"{short}_median"] = median
        params[f"{short}_mean"] = mean
        params[f"{short}_std"] = std

    # 0.5:0.5 form. Multiplying by two would not change rankings or labels.
    out["composite_score"] = 0.5 * out["z_war"] + 0.5 * out["z_util"]

    for outcome, score_col in (("composite", "composite_score"), ("war", cfg["war"])):
        train_score = out.loc[train_mask, score_col]

        cut2 = float(train_score.quantile(1 - cfg["top"]))
        q1 = float(train_score.quantile(1 / 3))
        q2 = float(train_score.quantile(2 / 3))

        out[f"{outcome}_2class"] = out[score_col].ge(cut2).astype(int)
        out[f"{outcome}_3class"] = np.select(
            [out[score_col].lt(q1), out[score_col].lt(q2)],
            [0, 1],
            default=2,
        ).astype(int)

        params[f"{outcome}_2class_cut"] = cut2
        params[f"{outcome}_3class_q1"] = q1
        params[f"{outcome}_3class_q2"] = q2

    return out, params


def reclassification_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    direction_rows = []
    pair_rows = []

    for role, cfg in ROLE_CONFIG.items():
        path = DATA_DIR / f"{role}_cohort_v2.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing cohort file: {path}")

        df = pd.read_csv(path)
        df, _ = prepare_outcomes(df, cfg)
        name_col = player_name_column(df)

        for subset_name, subset in (
            ("All", df),
            ("Test", df[df["split"].eq("test")].copy()),
        ):
            for task in ("2class", "3class"):
                war_col = f"war_{task}"
                comp_col = f"composite_{task}"

                changed = subset[war_col].ne(subset[comp_col])
                delta = subset[comp_col] - subset[war_col]

                summary_rows.append({
                    "Role": cfg["label"],
                    "Subset": subset_name,
                    "Task": task,
                    "N": len(subset),
                    "Agreement_N": int((~changed).sum()),
                    "Agreement_Rate": float((~changed).mean()),
                    "Reclassified_N": int(changed.sum()),
                    "Reclassified_Rate": float(changed.mean()),
                    "Composite_Up_N": int((delta > 0).sum()),
                    "Composite_Down_N": int((delta < 0).sum()),
                })

                changed_df = subset.loc[changed].copy()
                if len(changed_df):
                    changed_df["direction"] = np.where(
                        changed_df[comp_col] > changed_df[war_col],
                        "Composite higher class",
                        "Composite lower class",
                    )

                    for direction, group in changed_df.groupby("direction"):
                        direction_rows.append({
                            "Role": cfg["label"],
                            "Subset": subset_name,
                            "Task": task,
                            "Direction": direction,
                            "N": len(group),
                            "Mean_WAR": float(group[cfg["war"]].mean()),
                            "Median_WAR": float(group[cfg["war"]].median()),
                            f"Mean_{cfg['util_name']}": float(group[cfg["util"]].mean()),
                            f"Median_{cfg['util_name']}": float(group[cfg["util"]].median()),
                            "Mean_z_Utilization": float(group["z_util"].mean()),
                            "Median_z_Utilization": float(group["z_util"].median()),
                        })

        # Similar-WAR illustrative pairs:
        # Same WAR-only 2-class label, different Composite 2-class label,
        # and very similar 3-year WAR. The selection rule is fixed in advance:
        # choose the pair with the largest utilization difference.
        for subset_name, subset in (
            ("Test", df[df["split"].eq("test")].copy()),
            ("All", df.copy()),
        ):
            found = False

            for war_tolerance in (0.25, 0.50, 1.00):
                candidates = []

                local = subset[
                    [
                        name_col, "player_key", "fa_year",
                        cfg["war"], cfg["util"],
                        "z_war", "z_util", "composite_score",
                        "war_2class", "composite_2class",
                    ]
                ].dropna(subset=[cfg["war"], cfg["util"]]).copy()

                for i, j in combinations(local.index, 2):
                    a = local.loc[i]
                    b = local.loc[j]

                    if a["player_key"] == b["player_key"] and a["fa_year"] == b["fa_year"]:
                        continue
                    if int(a["war_2class"]) != int(b["war_2class"]):
                        continue
                    if int(a["composite_2class"]) == int(b["composite_2class"]):
                        continue

                    war_diff = abs(float(a[cfg["war"]]) - float(b[cfg["war"]]))
                    if war_diff > war_tolerance:
                        continue

                    util_diff = abs(float(a[cfg["util"]]) - float(b[cfg["util"]]))
                    candidates.append((util_diff, war_diff, a, b))

                if candidates:
                    candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
                    util_diff, war_diff, a, b = candidates[0]

                    # Put higher-utilization player first for easier reading.
                    if float(a[cfg["util"]]) < float(b[cfg["util"]]):
                        a, b = b, a

                    pair_rows.append({
                        "Role": cfg["label"],
                        "Subset": subset_name,
                        "WAR_Tolerance": war_tolerance,
                        "Player_A": a[name_col],
                        "FA_Year_A": int(a["fa_year"]),
                        "WAR_A": float(a[cfg["war"]]),
                        f"{cfg['util_name']}_A": float(a[cfg["util"]]),
                        "Composite_A": float(a["composite_score"]),
                        "WAR_2class_A": int(a["war_2class"]),
                        "Composite_2class_A": int(a["composite_2class"]),
                        "Player_B": b[name_col],
                        "FA_Year_B": int(b["fa_year"]),
                        "WAR_B": float(b[cfg["war"]]),
                        f"{cfg['util_name']}_B": float(b[cfg["util"]]),
                        "Composite_B": float(b["composite_score"]),
                        "WAR_2class_B": int(b["war_2class"]),
                        "Composite_2class_B": int(b["composite_2class"]),
                        "Absolute_WAR_Difference": float(war_diff),
                        f"Absolute_{cfg['util_name']}_Difference": float(util_diff),
                    })
                    found = True
                    break

            # Prefer test examples. Use All only if no test example exists.
            if subset_name == "Test" and found:
                break

    summary = pd.DataFrame(summary_rows)
    direction = pd.DataFrame(direction_rows)
    pairs = pd.DataFrame(pair_rows)

    summary.to_csv(
        OUTPUT_DIR / "table_reclassification_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    direction.to_csv(
        OUTPUT_DIR / "table_reclassification_direction.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pairs.to_csv(
        OUTPUT_DIR / "table_similar_war_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return summary, direction, pairs


def locate_prediction_files() -> tuple[Path, Path | None]:
    candidates = [
        ROOT / "outputs" / "temporal_accuracy_final",
        ROOT / "outputs" / "temporal_final",
    ]

    for directory in candidates:
        pred = directory / "temporal_predictions.csv"
        best = directory / "temporal_best_models.csv"
        if pred.exists():
            return pred, best if best.exists() else None

    raise FileNotFoundError(
        "temporal_predictions.csv was not found in "
        "outputs/temporal_accuracy_final or outputs/temporal_final."
    )


def selected_model(
    best: pd.DataFrame | None,
    role: str,
    outcome: str,
    task: str,
) -> str:
    # IMPORTANT:
    # Use the model specification reported in the paper.
    # Do not re-select models from temporal_best_models.csv here,
    # because this additional analysis must reproduce Table 6 exactly.
    return PAPER_MODEL_SPEC[(role, outcome, task)]


def bootstrap_accuracy_uncertainty() -> pd.DataFrame:
    prediction_path, best_path = locate_prediction_files()

    pred = pd.read_csv(prediction_path)
    best = pd.read_csv(best_path) if best_path is not None else None

    rng = np.random.default_rng(RANDOM_STATE)
    summary_rows = []
    raw_rows = []

    for role, cfg in ROLE_CONFIG.items():
        for task in ("2class", "3class"):
            comp_model = selected_model(best, role, "composite", task)
            war_model = selected_model(best, role, "war", task)

            comp = pred[
                pred["role"].eq(role)
                & pred["outcome"].eq("composite")
                & pred["task"].eq(task)
                & pred["model"].eq(comp_model)
                & pred["split"].eq("test")
            ][["player_key", "fa_year", "y_true", "y_pred"]].copy()

            war = pred[
                pred["role"].eq(role)
                & pred["outcome"].eq("war")
                & pred["task"].eq(task)
                & pred["model"].eq(war_model)
                & pred["split"].eq("test")
            ][["player_key", "fa_year", "y_true", "y_pred"]].copy()

            comp = comp.rename(columns={
                "y_true": "comp_true",
                "y_pred": "comp_pred",
            })
            war = war.rename(columns={
                "y_true": "war_true",
                "y_pred": "war_pred",
            })

            merged = comp.merge(
                war,
                on=["player_key", "fa_year"],
                how="inner",
                validate="one_to_one",
            )

            if len(merged) == 0:
                raise ValueError(f"No paired test predictions: {role}, {task}")

            comp_correct = (
                merged["comp_true"].to_numpy()
                == merged["comp_pred"].to_numpy()
            ).astype(float)

            war_correct = (
                merged["war_true"].to_numpy()
                == merged["war_pred"].to_numpy()
            ).astype(float)

            observed_comp = float(comp_correct.mean())
            observed_war = float(war_correct.mean())
            observed_diff = observed_comp - observed_war

            n = len(merged)
            bootstrap_diff = np.empty(N_BOOT, dtype=float)

            for b in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                bootstrap_diff[b] = (
                    comp_correct[idx].mean()
                    - war_correct[idx].mean()
                )

            boot_mean = float(bootstrap_diff.mean())
            boot_se = float(bootstrap_diff.std(ddof=1))
            ci_low, ci_high = np.percentile(bootstrap_diff, [2.5, 97.5])

            summary_rows.append({
                "Role": cfg["label"],
                "Task": task,
                "Composite_Model": comp_model,
                "WAR_Model": war_model,
                "N_Test": n,
                "Composite_Accuracy": observed_comp,
                "WAR_Only_Accuracy": observed_war,
                "Observed_Difference": observed_diff,
                "Bootstrap_Mean_Difference": boot_mean,
                "Bootstrap_SE": boot_se,
                "CI_2.5%": float(ci_low),
                "CI_97.5%": float(ci_high),
                "Bootstrap_Repetitions": N_BOOT,
            })

            raw_rows.extend({
                "Role": cfg["label"],
                "Task": task,
                "Iteration": b + 1,
                "Accuracy_Difference": float(value),
            } for b, value in enumerate(bootstrap_diff))

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_DIR / "table6_composite_vs_war_bootstrap_uncertainty.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(raw_rows).to_csv(
        OUTPUT_DIR / "bootstrap_accuracy_differences_10000.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return summary


def write_report(
    reclass: pd.DataFrame,
    direction: pd.DataFrame,
    pairs: pd.DataFrame,
    boot: pd.DataFrame,
) -> None:
    report_path = OUTPUT_DIR / "additional_composite_evidence_report.txt"

    lines = []
    lines.append("Composite vs WAR additional evidence")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1) Reclassification summary")
    lines.append(reclass.to_string(index=False))
    lines.append("")
    lines.append("2) Direction of reclassification")
    lines.append(direction.to_string(index=False))
    lines.append("")
    lines.append("3) Similar-WAR examples")
    lines.append(pairs.to_string(index=False))
    lines.append("")
    lines.append("4) Paired bootstrap uncertainty")
    lines.append(boot.to_string(index=False))
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    reclass, direction, pairs = reclassification_analysis()
    boot = bootstrap_accuracy_uncertainty()
    write_report(reclass, direction, pairs, boot)

    print("\nSaved:")
    for name in (
        "table_reclassification_summary.csv",
        "table_reclassification_direction.csv",
        "table_similar_war_examples.csv",
        "table6_composite_vs_war_bootstrap_uncertainty.csv",
        "bootstrap_accuracy_differences_10000.csv",
        "additional_composite_evidence_report.txt",
    ):
        print(" -", OUTPUT_DIR / name)

    print("\nBootstrap summary")
    print(
        boot[
            [
                "Role", "Task",
                "Composite_Accuracy", "WAR_Only_Accuracy",
                "Observed_Difference",
                "Bootstrap_Mean_Difference",
                "Bootstrap_SE",
                "CI_2.5%", "CI_97.5%",
            ]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
