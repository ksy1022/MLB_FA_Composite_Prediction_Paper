"""Compare version 2 cohorts with the legacy clean analysis files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data" / "processed_v2"
LEGACY = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "cohort_audit_v2"


def keys(frame: pd.DataFrame) -> pd.Series:
    return frame["player_key"].astype(str) + "|" + frame["anchor_year"].astype(int).astype(str)


def compare(
    name: str,
    legacy_path: Path,
    v2_path: Path,
    value_columns: list[str],
) -> list[dict[str, object]]:
    legacy = pd.read_csv(legacy_path)
    v2 = pd.read_csv(v2_path)
    legacy = legacy[legacy["anchor_year"].between(2012, 2021)].copy()
    legacy["contract_key"] = keys(legacy)
    v2["contract_key"] = keys(v2)

    legacy_keys = set(legacy["contract_key"])
    v2_keys = set(v2["contract_key"])
    only_legacy = legacy_keys - v2_keys
    only_v2 = v2_keys - legacy_keys
    common = legacy_keys & v2_keys

    detail = pd.concat(
        [
            legacy.loc[legacy["contract_key"].isin(only_legacy), ["contract_key", "player", "anchor_year"]]
            .assign(status="legacy_only"),
            v2.loc[v2["contract_key"].isin(only_v2), ["contract_key", "player", "anchor_year"]]
            .assign(status="v2_only"),
        ],
        ignore_index=True,
    )
    detail.to_csv(OUT / f"{name}_legacy_v2_membership_diff.csv", index=False, encoding="utf-8-sig")

    left = legacy.loc[legacy["contract_key"].isin(common), ["contract_key", *value_columns]]
    right = v2.loc[v2["contract_key"].isin(common), ["contract_key", *value_columns]]
    merged = left.merge(right, on="contract_key", suffixes=("_legacy", "_v2"))
    rows: list[dict[str, object]] = [
        {"dataset": name, "metric": "legacy_rows_2012_2021", "value": len(legacy)},
        {"dataset": name, "metric": "v2_rows_2012_2021", "value": len(v2)},
        {"dataset": name, "metric": "common_contracts", "value": len(common)},
        {"dataset": name, "metric": "legacy_only_contracts", "value": len(only_legacy)},
        {"dataset": name, "metric": "v2_only_contracts", "value": len(only_v2)},
    ]
    for column in value_columns:
        delta = (
            pd.to_numeric(merged[f"{column}_v2"], errors="coerce")
            - pd.to_numeric(merged[f"{column}_legacy"], errors="coerce")
        )
        rows.extend(
            [
                {
                    "dataset": name,
                    "metric": f"{column}_max_abs_diff_common",
                    "value": float(delta.abs().max()) if len(delta) else np.nan,
                },
                {
                    "dataset": name,
                    "metric": f"{column}_nonzero_diff_common",
                    "value": int(delta.abs().gt(1e-9).sum()),
                },
            ]
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = compare(
        "hitter",
        LEGACY / "final_hitter_fa_dataset_2011_2024_FAonly_clean_posgroup.csv",
        V2 / "hitter_cohort_v2.csv",
        ["pre_WAR_3y_sum", "post_WAR_3y_sum", "post_PA_3y"],
    )
    rows += compare(
        "pitcher",
        LEGACY / "final_pitcher_fa_dataset_2011_2024_FAonly_clean.csv",
        V2 / "pitcher_cohort_v2.csv",
        ["pre_WAR_3y_sum", "post_WAR_3y_sum", "post_IP_3y"],
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "legacy_v2_comparison.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
