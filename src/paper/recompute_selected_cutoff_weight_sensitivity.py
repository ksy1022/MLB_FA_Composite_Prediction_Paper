"""Recompute Composite weight sensitivity after the role-specific 2-class cutoff changes.

Only the 2-class label-agreement portion is affected by the newly selected High
proportion. The script also recomputes the 3-class agreement for completeness.

Weights compared:
- 0.5 WAR + 0.5 Utilization (baseline)
- 0.4 WAR + 0.6 Utilization
- 0.6 WAR + 0.4 Utilization

For every weight, the numerical label boundaries are estimated on the training
partition only and then applied unchanged to validation/test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
MODELING = ROOT / "src" / "modeling"
sys.path.insert(0, str(MODELING))
from optimized_2class_common import ROLE_CONFIG, load_role_frame, prepare_scores  # noqa: E402

OUT = ROOT / "outputs" / "optimized_2class"
CONFIG_PATH = OUT / "selected_role_config.json"
WEIGHTS = [(0.5, 0.5), (0.4, 0.6), (0.6, 0.4)]


def labels_for_weight(df: pd.DataFrame, top_ratio: float, war_weight: float, util_weight: float):
    score = war_weight * df["z_war"] + util_weight * df["z_playtime"]
    train_mask = df["split"].eq("train")
    train_score = score.loc[train_mask]

    binary_cut = float(train_score.quantile(1.0 - top_ratio))
    low_cut = float(train_score.quantile(1.0 / 3.0))
    high_cut = float(train_score.quantile(2.0 / 3.0))

    binary = score.ge(binary_cut).astype(int)
    multi = np.select(
        [score.lt(low_cut), score.lt(high_cut)], [0, 1], default=2
    ).astype(int)
    return score, binary, multi, binary_cut, low_cut, high_cut


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Run run_optimized_2class.py first: {CONFIG_PATH}")
    selected = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    agreement_rows = []
    parameter_rows = []

    for role, cfg in ROLE_CONFIG.items():
        raw = load_role_frame(role)
        scored, _ = prepare_scores(raw, cfg)
        top_ratio = float(selected["roles"][role]["selected_top_ratio"])

        labels = {}
        for war_w, util_w in WEIGHTS:
            key = f"{war_w:.1f}_{util_w:.1f}"
            score, binary, multi, binary_cut, low_cut, high_cut = labels_for_weight(
                scored, top_ratio, war_w, util_w
            )
            labels[key] = {"score": score, "binary": binary, "multi": multi}
            parameter_rows.append(
                {
                    "role": role,
                    "selected_top_ratio": top_ratio,
                    "war_weight": war_w,
                    "util_weight": util_w,
                    "binary_cut": binary_cut,
                    "threeclass_low_cut": low_cut,
                    "threeclass_high_cut": high_cut,
                }
            )

        base = labels["0.5_0.5"]
        for alt_key, (war_w, util_w) in {
            "0.4_0.6": (0.4, 0.6),
            "0.6_0.4": (0.6, 0.4),
        }.items():
            alt = labels[alt_key]
            for split_name in ("all", "train", "valid", "test"):
                mask = (
                    pd.Series(True, index=scored.index)
                    if split_name == "all"
                    else scored["split"].eq(split_name)
                )
                for task, base_label, alt_label in (
                    ("2class", base["binary"], alt["binary"]),
                    ("3class", base["multi"], alt["multi"]),
                ):
                    a = np.asarray(base_label)[np.asarray(mask)]
                    b = np.asarray(alt_label)[np.asarray(mask)]
                    agreement_rows.append(
                        {
                            "role": role,
                            "selected_top_ratio": top_ratio,
                            "task": task,
                            "split": split_name,
                            "war_weight": war_w,
                            "util_weight": util_w,
                            "n": int(len(a)),
                            "label_agreement": float(np.mean(a == b)),
                            "cohen_kappa": float(cohen_kappa_score(a, b)),
                        }
                    )

    agreement = pd.DataFrame(agreement_rows)
    params = pd.DataFrame(parameter_rows)
    agreement.to_csv(
        OUT / "selected_cutoff_weight_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    params.to_csv(
        OUT / "selected_cutoff_weight_sensitivity_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        agreement[agreement["split"].eq("all")][
            [
                "role", "selected_top_ratio", "task", "war_weight", "util_weight",
                "label_agreement", "cohen_kappa",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
