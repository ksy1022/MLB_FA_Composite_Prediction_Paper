"""Create player-grouped splits and train-fitted Composite labels.

No model is trained here. The generated split files are the single source of
truth for subsequent classification and regression experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed_v2"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "leakage_safe_splits"


def grouped_split(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    remain_idx, test_idx = next(outer.split(df, groups=df["player_key"]))
    remain = df.iloc[remain_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_local, valid_local = next(
        inner.split(remain, groups=remain["player_key"])
    )
    split = pd.Series(index=df.index, dtype="object")
    split.iloc[test_idx] = "test"
    split.iloc[remain_idx[train_local]] = "train"
    split.iloc[remain_idx[valid_local]] = "valid"
    return split


def fit_composite(
    df: pd.DataFrame,
    war_col: str,
    playtime_col: str,
    success_top_ratio: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    train = out[out["split"].eq("train")]
    parameters: dict[str, float] = {}
    for col, short in [(war_col, "war"), (playtime_col, "playtime")]:
        median = float(train[col].median())
        filled_train = train[col].fillna(median)
        mean = float(filled_train.mean())
        std = float(filled_train.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            raise ValueError(f"Training standard deviation is invalid for {col}: {std}")
        parameters[f"{short}_median"] = median
        parameters[f"{short}_mean"] = mean
        parameters[f"{short}_std"] = std
        out[f"z_post_{short}"] = (out[col].fillna(median) - mean) / std

    out["composite_score"] = out["z_post_war"] + out["z_post_playtime"]
    train_score = out.loc[out["split"].eq("train"), "composite_score"]
    cut_2class = float(train_score.quantile(1 - success_top_ratio))
    cut_3_low = float(train_score.quantile(1 / 3))
    cut_3_high = float(train_score.quantile(2 / 3))
    parameters.update(
        {
            "success_top_ratio": success_top_ratio,
            "cut_2class": cut_2class,
            "cut_3class_low": cut_3_low,
            "cut_3class_high": cut_3_high,
        }
    )
    out["success_2class"] = out["composite_score"].ge(cut_2class).astype(int)
    out["success_3class"] = np.select(
        [
            out["composite_score"].lt(cut_3_low),
            out["composite_score"].lt(cut_3_high),
        ],
        [0, 1],
        default=2,
    ).astype(int)
    return out, parameters


def validate_no_group_leakage(df: pd.DataFrame) -> None:
    sets = {
        name: set(part["player_key"])
        for name, part in df.groupby("split", observed=True)
    }
    names = sorted(sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(f"{left}/{right} share {len(overlap)} players")


def prepare(name: str, war_col: str, playtime_col: str, ratio: float, seed: int) -> None:
    path = DATA_DIR / f"{name}_cohort_v2.csv"
    df = pd.read_csv(path)
    df["split"] = grouped_split(df, seed)
    validate_no_group_leakage(df)
    labeled, params = fit_composite(df, war_col, playtime_col, ratio)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(OUTPUT_DIR / f"{name}_labeled_splits.csv", index=False, encoding="utf-8-sig")
    params.update(
        {
            "dataset": name,
            "random_seed": seed,
            "n_rows": len(labeled),
            "n_players": int(labeled["player_key"].nunique()),
            "split_rows": labeled["split"].value_counts().to_dict(),
            "split_players": labeled.groupby("split")["player_key"].nunique().to_dict(),
        }
    )
    with (OUTPUT_DIR / f"{name}_label_parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(params, handle, ensure_ascii=False, indent=2)


def main(seed: int) -> None:
    prepare("hitter", "post_WAR_3y_sum", "post_PA_3y", 0.40, seed)
    prepare("sp", "post_WAR_3y_sum", "post_IP_3y", 0.20, seed)
    prepare("rp", "post_WAR_3y_sum", "post_IP_3y", 0.20, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args().seed)
