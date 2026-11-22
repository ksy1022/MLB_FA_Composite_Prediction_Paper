import pandas as pd
import numpy as np
from pathlib import Path
import re

def year_from_filename(path):
    m = re.search(r"(\d{4})", path.name)
    if not m:
        raise ValueError(f"연도 없음: {path.name}")
    return int(m.group(1))

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BATTING_DIR = PROJECT_ROOT / "data" / "raw" / "batting"
PITCHING_DIR = PROJECT_ROOT / "data" / "raw" / "pitching"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "player_windows"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def load_mlb_seasons(data_dir, prefix):
    files = sorted(data_dir.glob(f"{prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"{prefix} 파일 없음")

    rows = []

    for f in files:
        season = year_from_filename(f)
        df = pd.read_csv(f)
        df["season"] = season

        # 고유 선수 키
        if "Player-additional" in df.columns:
            df["player_key"] = df["Player-additional"]
        else:
            df["player_key"] = df["Player"]

        # 트레이드 처리:
        # TOT / 2TM / 3TM 행을 우선 사용
        df["_is_total"] = (
            df["Team"].astype(str).str.match(r"^\d+TM$")
            | (df["Team"] == "TOT")
        ).astype(int)

        df = (
            df.sort_values(["player_key", "_is_total"], ascending=[True, False])
              .drop_duplicates(subset=["player_key"], keep="first")
              .drop(columns="_is_total")
        )

        rows.append(df)

    return pd.concat(rows, ignore_index=True)

bat_all = load_mlb_seasons(
    BATTING_DIR,
    "mlb_batting_"
)

pit_all = load_mlb_seasons(
    PITCHING_DIR,
    "mlb_pitching_"
)

print("Bat seasons:", bat_all["season"].min(), "-", bat_all["season"].max(), "rows:", len(bat_all))
print("Pit seasons:", pit_all["season"].min(), "-", pit_all["season"].max(), "rows:", len(pit_all))


def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def make_hitter_fa_windows(bat):
    for c in ["Age", "WAR", "PA", "BB", "SO", "OPS+"]:
        if c in bat.columns:
            bat[c] = to_num(bat[c])

    bat["BB_rate"] = bat["BB"] / bat["PA"]
    bat["K_rate"] = bat["SO"] / bat["PA"]

    seasons = sorted(bat["season"].unique())
    rows = []

    for y in seasons:
        pre_years = [y-3, y-2, y-1]
        post_years = [y, y+1, y+2]

        if not set(pre_years).issubset(seasons):
            continue
        if not set(post_years).issubset(seasons):
            continue

        pre = bat[bat["season"].isin(pre_years)]
        post = bat[bat["season"].isin(post_years)]

        pre_g = pre.groupby("player_key", as_index=False).agg(
            player=("Player", "last"),
            age_at_contract=("Age", "last"),
            pre_PA_3y=("PA", "sum"),
            pre_WAR_3y_sum=("WAR", "sum"),
            pre_WAR_3y_avg=("WAR", "mean"),
            pre_WAR_3y_std=("WAR", "std"),
            pre_OPSplus_3y_avg=("OPS+", "mean"),
            pre_BB_rate_3y_avg=("BB_rate", "mean"),
            pre_K_rate_3y_avg=("K_rate", "mean"),
        )

        pre_g["pre_WAR_3y_std"] = pre_g["pre_WAR_3y_std"].fillna(0)

        post_g = post.groupby("player_key", as_index=False).agg(
            post_WAR_3y_sum=("WAR", "sum"),
            post_PA_3y=("PA", "sum"),
        )

        out = pre_g.merge(post_g, on="player_key", how="inner")
        out["anchor_year"] = y
        rows.append(out)

    return pd.concat(rows, ignore_index=True)

hitter_ds = make_hitter_fa_windows(bat_all)
print("Hitter dataset:", hitter_ds.shape)

HITTER_OUT = (
    INTERIM_DIR / "mlb_hitter_fa_windows.csv"
)

hitter_ds.to_csv(
    HITTER_OUT,
    index=False,
    encoding="utf-8-sig"
)

print(f"[OK] Saved: {HITTER_OUT}")


def make_pitcher_fa_windows(pit):
    for c in ["Age", "WAR", "IP", "ERA", "FIP", "BB", "SO", "HR", "GS", "G"]:
        if c in pit.columns:
            pit[c] = to_num(pit[c])

    pit["KBB"] = pit["SO"] / pit["BB"].replace(0, np.nan)

    seasons = sorted(pit["season"].unique())
    rows = []

    for y in seasons:
        pre_years = [y-3, y-2, y-1]
        post_years = [y, y+1, y+2]

        if not set(pre_years).issubset(seasons):
            continue
        if not set(post_years).issubset(seasons):
            continue

        pre = pit[pit["season"].isin(pre_years)]
        post = pit[pit["season"].isin(post_years)]

        pre_g = pre.groupby("player_key", as_index=False).agg(
            player=("Player", "last"),
            age_at_contract=("Age", "last"),
            pre_IP_3y=("IP", "sum"),
            pre_WAR_3y_sum=("WAR", "sum"),
            pre_WAR_3y_avg=("WAR", "mean"),
            pre_WAR_3y_std=("WAR", "std"),
            pre_ERA_3y_avg=("ERA", "mean"),
            pre_FIP_3y_avg=("FIP", "mean"),
            pre_KBB_3y_avg=("KBB", "mean"),
            pre_GS_3y_sum=("GS", "sum"),
            pre_G_3y_sum=("G", "sum"),
        )
        pre_g["pre_WAR_3y_std"] = pre_g["pre_WAR_3y_std"].fillna(0)

        post_g = post.groupby("player_key", as_index=False).agg(
            post_WAR_3y_sum=("WAR", "sum"),
            post_IP_3y=("IP", "sum"),
        )

        out = pre_g.merge(post_g, on="player_key", how="inner")
        out["anchor_year"] = y
        rows.append(out)

    return pd.concat(rows, ignore_index=True)

pitcher_ds = make_pitcher_fa_windows(pit_all)
print("Pitcher dataset:", pitcher_ds.shape)
PITCHER_OUT = (
    INTERIM_DIR / "mlb_pitcher_fa_windows.csv"
)

pitcher_ds.to_csv(
    PITCHER_OUT,
    index=False,
    encoding="utf-8-sig"
)

print(f"[OK] Saved: {PITCHER_OUT}")
