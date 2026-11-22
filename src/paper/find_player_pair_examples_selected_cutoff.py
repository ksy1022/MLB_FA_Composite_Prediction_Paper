"""Find updated same-WAR / different-utilization examples after cutoff reselection.

For each role, the script searches 2012-2021 contracts for a pair satisfying:
- absolute 3-year WAR difference <= 0.25
- same WAR-only 2-class group
- different Composite 2-class group

Among eligible pairs it chooses the pair with the largest PA/IP difference.
This regenerates the paper's illustrative player-pair table when the 2-class
cutoff changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELING = ROOT / "src" / "modeling"
sys.path.insert(0, str(MODELING))
from optimized_2class_common import (  # noqa: E402
    ROLE_CONFIG,
    ROLE_LABELS,
    add_binary_labels,
    load_role_frame,
    prepare_scores,
)

OUT = ROOT / "outputs" / "optimized_2class"
CONFIG_PATH = OUT / "selected_role_config.json"
MAX_WAR_DIFF = 0.25


def display_name(row: pd.Series) -> str:
    for col in ("player", "Player", "Name"):
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return str(row["player_key"])


def best_pair(role: str, df: pd.DataFrame, cfg: dict) -> list[dict]:
    war_col = cfg["war"]
    play_col = cfg["playtime"]
    rows = df.reset_index(drop=True)
    best = None

    for i in range(len(rows)):
        a = rows.iloc[i]
        for j in range(i + 1, len(rows)):
            b = rows.iloc[j]
            if pd.isna(a[war_col]) or pd.isna(b[war_col]):
                continue
            war_diff = abs(float(a[war_col]) - float(b[war_col]))
            if war_diff > MAX_WAR_DIFF:
                continue
            if int(a["war_2class"]) != int(b["war_2class"]):
                continue
            if int(a["composite_2class"]) == int(b["composite_2class"]):
                continue
            if pd.isna(a[play_col]) or pd.isna(b[play_col]):
                continue
            play_diff = abs(float(a[play_col]) - float(b[play_col]))
            candidate = (play_diff, -war_diff, i, j)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return []

    play_diff, neg_war_diff, i, j = best
    a, b = rows.iloc[i], rows.iloc[j]
    pair_id = f"{role}_1"
    result = []
    for row in (a, b):
        result.append(
            {
                "pair_id": pair_id,
                "role": role,
                "role_label": ROLE_LABELS[role],
                "player": display_name(row),
                "player_key": row["player_key"],
                "fa_year": int(row["fa_year"]),
                "post_3y_WAR": float(row[war_col]),
                "post_3y_utilization": float(row[play_col]),
                "WAR_only_group": "High" if int(row["war_2class"]) == 1 else "Other",
                "Composite_score": float(row["composite_score"]),
                "Composite_group": "High" if int(row["composite_2class"]) == 1 else "Other",
                "pair_WAR_difference": float(-neg_war_diff),
                "pair_utilization_difference": float(play_diff),
            }
        )
    return result


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Run run_optimized_2class.py first: {CONFIG_PATH}")
    selected = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    all_rows: list[dict] = []
    for role, cfg in ROLE_CONFIG.items():
        raw = load_role_frame(role)
        scored, _ = prepare_scores(raw, cfg)
        top_ratio = float(selected["roles"][role]["selected_top_ratio"])
        labeled, _ = add_binary_labels(scored, cfg, top_ratio)
        all_rows.extend(best_pair(role, labeled, cfg))

    result = pd.DataFrame(all_rows)
    result.to_csv(
        OUT / "updated_player_pair_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if result.empty:
        print("No pair satisfying the requested criteria was found.")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
