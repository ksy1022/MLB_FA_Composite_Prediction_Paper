from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_FA_DIR = PROJECT_ROOT / "data" / "interim" / "fa_contracts"
REGISTER_DIR = PROJECT_ROOT / "data" / "external" / "register-master" / "data"

FA_EVENTS = INTERIM_FA_DIR / "mlb_fa_events_2007_2024.csv"
FA_LIST_OUTPUT = INTERIM_FA_DIR / "mlb_fa_list_2007_2024.csv"

# 1) FA 이벤트 로드
fa = pd.read_csv(FA_EVENTS)

# player_mlbam dtype 강제
fa["player_mlbam"] = pd.to_numeric(fa["player_mlbam"], errors="coerce").astype("Int64")

# 2) Chadwick people 파일들 합치기
people_files = sorted(REGISTER_DIR.glob("people-*.csv"))
if not people_files:
    raise FileNotFoundError(f"people-*.csv를 못 찾음: {REGISTER_DIR.resolve()}")

# low_memory=False로 DtypeWarning 줄이기
people = pd.concat((pd.read_csv(p, low_memory=False) for p in people_files), ignore_index=True)

# 필요한 키만
keep_cols = ["key_mlbam", "key_bbref", "name_first", "name_last"]
missing = [c for c in keep_cols if c not in people.columns]
if missing:
    raise RuntimeError(f"register 컬럼이 예상과 달라요. 없는 컬럼: {missing}\n현재 컬럼 예: {people.columns.tolist()[:60]}")

keep = people[keep_cols].copy()

# key_mlbam도 dtype 강제
keep["key_mlbam"] = pd.to_numeric(keep["key_mlbam"], errors="coerce").astype("Int64")
keep["key_bbref"] = keep["key_bbref"].astype(str)
keep.loc[keep["key_bbref"].isin(["", "nan", "None"]), "key_bbref"] = pd.NA

# 3) merge
out = fa.merge(keep, left_on="player_mlbam", right_on="key_mlbam", how="left")

# 4) 매칭률 확인
match_rate = out["key_bbref"].notna().mean()
print(f"[INFO] match rate (bbref found): {match_rate:.3f}")

# 5) 모델용 FA 리스트
fa_list = (
    out.dropna(subset=["key_bbref"])
       .drop_duplicates(subset=["fa_year", "key_bbref"])
       .rename(columns={"key_bbref": "player_key"})
       [["fa_year", "player_key", "player_mlbam", "type_desc", "description"]]
)

# 6) 저장
INTERIM_FA_DIR.mkdir(parents=True, exist_ok=True)

fa_list.to_csv(
    FA_LIST_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print(f"[OK] Saved: {FA_LIST_OUTPUT}")
print("FA list rows:", len(fa_list))