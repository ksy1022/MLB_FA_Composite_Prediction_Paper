"""Build an auditable FA analysis cohort from contracts first.

This pipeline fixes the survivor bias in the original implementation:

1. Start from the Spotrac contract table.
2. Resolve a Baseball-Reference player id from the three pre-contract seasons.
3. Create all six required season slots around the contract.
4. Treat an absent post-contract MLB row as a real zero only after the player
   identity and the source season file have both been verified.
5. Preserve matching failures in an exclusion report instead of silently
   dropping them.

The original processed files are never overwritten.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_BATTING = PROJECT_ROOT / "data" / "raw" / "batting"
RAW_PITCHING = PROJECT_ROOT / "data" / "raw" / "pitching"
CONTRACTS = PROJECT_ROOT / "data" / "processed" / "spotrac_fa_2011_2024_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_v2"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "cohort_audit_v2"


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[*#]", "", text)
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def season_from_path(path: Path) -> int:
    match = re.search(r"(\d{4})", path.name)
    if not match:
        raise ValueError(f"Cannot identify season from {path.name}")
    return int(match.group(1))


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_seasons(directory: Path, prefix: str) -> tuple[pd.DataFrame, set[int]]:
    frames: list[pd.DataFrame] = []
    source_years: set[int] = set()
    for path in sorted(directory.glob(f"{prefix}*.csv")):
        season = season_from_path(path)
        source_years.add(season)
        frame = pd.read_csv(path)
        frame.columns = frame.columns.astype(str).str.strip()
        frame["season"] = season
        if "Player-additional" not in frame:
            raise ValueError(f"Player-additional is missing from {path}")
        frame["player_key"] = frame["Player-additional"].astype(str).str.strip()
        frame["player_norm"] = frame["Player"].map(normalize_name)
        frame = frame[~frame["player_key"].isin(["", "nan", "-9999"])].copy()

        is_total = (
            frame["Team"].astype(str).str.match(r"^\d+TM$")
            | frame["Team"].astype(str).eq("TOT")
        )
        frame["_is_total"] = is_total.astype(int)
        frame = (
            frame.sort_values(["player_key", "_is_total"], ascending=[True, False])
            .drop_duplicates(["player_key"], keep="first")
            .drop(columns="_is_total")
        )
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No files matched {directory / (prefix + '*.csv')}")
    return pd.concat(frames, ignore_index=True), source_years


def classify_contract_position(pos: object) -> str:
    tokens = {x for x in re.split(r"[/,\s-]+", str(pos).upper()) if x}
    if tokens & {"SP", "RP", "P", "CL"}:
        return "pitcher"
    return "hitter"


def pitcher_role(pos: object, pre_rows: pd.DataFrame) -> str:
    tokens = {x for x in re.split(r"[/,\s-]+", str(pos).upper()) if x}
    if "SP" in tokens:
        return "SP"
    if "RP" in tokens or "CL" in tokens:
        return "RP"
    gs = numeric(pre_rows.get("GS", pd.Series(dtype=float))).sum()
    games = numeric(pre_rows.get("G", pd.Series(dtype=float))).sum()
    return "SP" if games > 0 and gs / games >= 0.5 else "RP"


def deduplicate_contracts(contracts: pd.DataFrame) -> pd.DataFrame:
    out = contracts.copy()
    out["player_norm"] = out["Player"].map(normalize_name)
    out["fa_year"] = numeric(out["fa_year"]).astype("Int64")
    out["Value"] = numeric(out["Value"])
    out["Yrs"] = numeric(out["Yrs"])
    out["AAV"] = numeric(out["AAV"])
    out["_source_row"] = np.arange(len(out))
    return (
        out.sort_values(
            ["player_norm", "fa_year", "Value", "_source_row"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["player_norm", "fa_year"], keep="first")
        .reset_index(drop=True)
    )


def resolve_player(
    contract: pd.Series, seasons: pd.DataFrame
) -> tuple[str | None, str, int]:
    year = int(contract["fa_year"])
    candidates = seasons[
        seasons["season"].between(year - 3, year - 1)
        & seasons["player_norm"].eq(contract["player_norm"])
    ]
    keys = sorted(candidates["player_key"].dropna().astype(str).unique())
    if len(keys) == 1:
        return keys[0], "matched_pre_name", len(keys)
    if not keys:
        return None, "no_pre_match", 0
    return None, "ambiguous_pre_match", len(keys)


def safe_mean(frame: pd.DataFrame, column: str) -> float:
    return numeric(frame[column]).mean() if column in frame else np.nan


def safe_sum(frame: pd.DataFrame, column: str) -> float:
    return numeric(frame[column]).fillna(0).sum() if column in frame else np.nan


def build_hitter_row(
    contract: pd.Series, key: str, seasons: pd.DataFrame, source_years: set[int]
) -> dict[str, object]:
    year = int(contract["fa_year"])
    player_rows = seasons[seasons["player_key"].eq(key)].copy()
    pre = player_rows[player_rows["season"].between(year - 3, year - 1)].sort_values("season")
    post = player_rows[player_rows["season"].between(year, year + 2)].copy()
    for col in ["Age", "WAR", "PA", "BB", "SO", "OPS+"]:
        if col in pre:
            pre[col] = numeric(pre[col])
        if col in post:
            post[col] = numeric(post[col])
    pre["BB_rate"] = pre["BB"] / pre["PA"].replace(0, np.nan)
    pre["K_rate"] = pre["SO"] / pre["PA"].replace(0, np.nan)
    post_years_present = set(post["season"].astype(int))
    return {
        "player_key": key,
        "player": pre["Player"].iloc[-1],
        "age_at_contract": pre["Age"].iloc[-1],
        "pre_seasons_with_MLB_record": pre["season"].nunique(),
        "post_seasons_with_MLB_record": post["season"].nunique(),
        "post_zero_filled_seasons": sum(
            season in source_years and season not in post_years_present
            for season in range(year, year + 3)
        ),
        "pre_PA_3y": safe_sum(pre, "PA"),
        "pre_WAR_3y_sum": safe_sum(pre, "WAR"),
        "pre_WAR_3y_avg": safe_mean(pre, "WAR"),
        "pre_WAR_3y_std": numeric(pre["WAR"]).std() if "WAR" in pre else np.nan,
        "pre_OPSplus_3y_avg": safe_mean(pre, "OPS+"),
        "pre_BB_rate_3y_avg": safe_mean(pre, "BB_rate"),
        "pre_K_rate_3y_avg": safe_mean(pre, "K_rate"),
        "post_WAR_3y_sum": safe_sum(post, "WAR"),
        "post_PA_3y": safe_sum(post, "PA"),
    }


def build_pitcher_row(
    contract: pd.Series, key: str, seasons: pd.DataFrame, source_years: set[int]
) -> dict[str, object]:
    year = int(contract["fa_year"])
    player_rows = seasons[seasons["player_key"].eq(key)].copy()
    pre = player_rows[player_rows["season"].between(year - 3, year - 1)].sort_values("season")
    post = player_rows[player_rows["season"].between(year, year + 2)].copy()
    for col in ["Age", "WAR", "IP", "ERA", "FIP", "BB", "SO", "GS", "G"]:
        if col in pre:
            pre[col] = numeric(pre[col])
        if col in post:
            post[col] = numeric(post[col])
    pre["KBB"] = pre["SO"] / pre["BB"].replace(0, np.nan)
    post_years_present = set(post["season"].astype(int))
    return {
        "player_key": key,
        "player": pre["Player"].iloc[-1],
        "role": pitcher_role(contract["Pos"], pre),
        "age_at_contract": pre["Age"].iloc[-1],
        "pre_seasons_with_MLB_record": pre["season"].nunique(),
        "post_seasons_with_MLB_record": post["season"].nunique(),
        "post_zero_filled_seasons": sum(
            season in source_years and season not in post_years_present
            for season in range(year, year + 3)
        ),
        "pre_IP_3y": safe_sum(pre, "IP"),
        "pre_WAR_3y_sum": safe_sum(pre, "WAR"),
        "pre_WAR_3y_avg": safe_mean(pre, "WAR"),
        "pre_WAR_3y_std": numeric(pre["WAR"]).std() if "WAR" in pre else np.nan,
        "pre_ERA_3y_avg": safe_mean(pre, "ERA"),
        "pre_FIP_3y_avg": safe_mean(pre, "FIP"),
        "pre_KBB_3y_avg": safe_mean(pre, "KBB"),
        "pre_GS_3y_sum": safe_sum(pre, "GS"),
        "pre_G_3y_sum": safe_sum(pre, "G"),
        "post_WAR_3y_sum": safe_sum(post, "WAR"),
        "post_IP_3y": safe_sum(post, "IP"),
    }


def contract_fields(contract: pd.Series) -> dict[str, object]:
    return {
        "anchor_year": int(contract["fa_year"]),
        "player_norm": contract["player_norm"],
        "fa_year": int(contract["fa_year"]),
        "Pos": contract["Pos"],
        "Yrs": contract["Yrs"],
        "Value": contract["Value"],
        "AAV": contract["AAV"],
        "is_real_fa": 1,
    }


def build_cohort(start_year: int, end_year: int) -> None:
    contracts = deduplicate_contracts(pd.read_csv(CONTRACTS))
    contracts = contracts[contracts["fa_year"].between(start_year, end_year)].copy()
    contracts["record_type"] = contracts["Pos"].map(classify_contract_position)

    batting, batting_years = load_seasons(RAW_BATTING, "mlb_batting_")
    pitching, pitching_years = load_seasons(RAW_PITCHING, "mlb_pitching_")

    output_rows: dict[str, list[dict[str, object]]] = {"hitter": [], "pitcher": []}
    audit_rows: list[dict[str, object]] = []

    for _, contract in contracts.iterrows():
        kind = contract["record_type"]
        source = batting if kind == "hitter" else pitching
        source_years = batting_years if kind == "hitter" else pitching_years
        key, status, candidate_count = resolve_player(contract, source)
        required_years = set(range(int(contract["fa_year"]) - 3, int(contract["fa_year"]) + 3))
        missing_source_years = sorted(required_years - source_years)

        audit = {
            "Player": contract["Player"],
            "player_norm": contract["player_norm"],
            "fa_year": int(contract["fa_year"]),
            "Pos": contract["Pos"],
            "record_type": kind,
            "player_key": key,
            "match_status": status,
            "candidate_key_count": candidate_count,
            "missing_source_years": ",".join(map(str, missing_source_years)),
            "included": False,
            "exclusion_reason": "",
        }
        if missing_source_years:
            audit["exclusion_reason"] = "source_season_missing"
        elif key is None:
            audit["exclusion_reason"] = status
        else:
            builder = build_hitter_row if kind == "hitter" else build_pitcher_row
            built = builder(contract, key, source, source_years)
            built.update(contract_fields(contract))
            output_rows[kind].append(built)
            audit["included"] = True
        audit_rows.append(audit)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    hitter = pd.DataFrame(output_rows["hitter"])
    pitcher = pd.DataFrame(output_rows["pitcher"])
    audit = pd.DataFrame(audit_rows)
    hitter["Pos_group"] = hitter["Pos"].map(hitter_position_group)

    hitter.to_csv(OUTPUT_DIR / "hitter_cohort_v2.csv", index=False, encoding="utf-8-sig")
    pitcher.to_csv(OUTPUT_DIR / "pitcher_cohort_v2.csv", index=False, encoding="utf-8-sig")
    pitcher[pitcher["role"].eq("SP")].to_csv(
        OUTPUT_DIR / "sp_cohort_v2.csv", index=False, encoding="utf-8-sig"
    )
    pitcher[pitcher["role"].eq("RP")].to_csv(
        OUTPUT_DIR / "rp_cohort_v2.csv", index=False, encoding="utf-8-sig"
    )
    audit.to_csv(AUDIT_DIR / "contract_level_audit.csv", index=False, encoding="utf-8-sig")

    summary = make_summary(contracts, hitter, pitcher, audit, start_year, end_year)
    summary.to_csv(AUDIT_DIR / "cohort_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


def hitter_position_group(pos: object) -> str:
    tokens = {x for x in re.split(r"[/,\s-]+", str(pos).upper()) if x}
    if "C" in tokens:
        return "C"
    if tokens & {"1B", "2B", "3B", "SS"}:
        return "IF"
    if tokens & {"LF", "CF", "RF", "OF"}:
        return "OF"
    if "DH" in tokens:
        return "DH"
    return "Other"


def make_summary(
    contracts: pd.DataFrame,
    hitter: pd.DataFrame,
    pitcher: pd.DataFrame,
    audit: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    rows = [
        ("study_year_start", start_year),
        ("study_year_end", end_year),
        ("deduplicated_contracts", len(contracts)),
        ("included_contracts", int(audit["included"].sum())),
        ("excluded_contracts", int((~audit["included"]).sum())),
        ("hitter_contracts", len(hitter)),
        ("pitcher_contracts", len(pitcher)),
        ("sp_contracts", int(pitcher["role"].eq("SP").sum())),
        ("rp_contracts", int(pitcher["role"].eq("RP").sum())),
        ("unique_hitters", hitter["player_key"].nunique()),
        ("unique_pitchers", pitcher["player_key"].nunique()),
        ("hitter_zero_filled_post_seasons", int(hitter["post_zero_filled_seasons"].sum())),
        ("pitcher_zero_filled_post_seasons", int(pitcher["post_zero_filled_seasons"].sum())),
        ("hitter_all_three_post_seasons_absent", int(hitter["post_seasons_with_MLB_record"].eq(0).sum())),
        ("pitcher_all_three_post_seasons_absent", int(pitcher["post_seasons_with_MLB_record"].eq(0).sum())),
    ]
    for reason, count in audit.loc[~audit["included"], "exclusion_reason"].value_counts().items():
        rows.append((f"excluded_{reason}", int(count)))
    return pd.DataFrame(rows, columns=["metric", "value"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2021)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_cohort(args.start_year, args.end_year)
