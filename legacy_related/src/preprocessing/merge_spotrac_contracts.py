import pandas as pd
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT / "data" / "raw" / "spotrac_fa"
)

INTERIM_FA_DIR = (
    PROJECT_ROOT / "data" / "interim" / "fa_contracts"
)

OUTPUT_PATH = (
    INTERIM_FA_DIR / "spotrac_fa_2011_2024.csv"
)

INTERIM_FA_DIR.mkdir(parents=True, exist_ok=True)

dfs = []
for p in DATA_DIR.glob("spotrac_fa_*.csv"):
    m = re.search(r"(\d{4})", p.name)
    if not m:
        continue
    year = int(m.group(1))

    df = pd.read_csv(p)
    df["fa_year"] = year
    dfs.append(df)

fa_all = pd.concat(dfs, ignore_index=True)

# ---- 안전 숫자 변환 함수 ----
def to_num(s: pd.Series) -> pd.Series:
    # $ , 공백, 괄호 각주 (1) 같은 거 제거하고 숫자/소수점만 남김
    s = s.astype(str)
    s = s.str.replace(r"\(.*?\)", "", regex=True)          # (1) 같은 각주 제거
    s = s.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)        # 숫자/소수점/마이너스만 남김
    return pd.to_numeric(s, errors="coerce")

# 숫자형 정리
for c in ["Yrs", "Value", "AAV"]:
    if c in fa_all.columns:
        fa_all[c] = to_num(fa_all[c])

# Yrs가 0이거나 NaN이면 AAV 계산이 이상해지니 정리 (선택)
# fa_all = fa_all[fa_all["Yrs"].fillna(0) > 0].copy()

# 저장 (너는 2011~2024 다 했으니까 파일명도 그렇게)
fa_all.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(f"[OK] Saved: {OUTPUT_PATH}")
print(fa_all[["fa_year","Player","Pos","Yrs","Value","AAV"]].head())
print("NaN rates:", fa_all[["Yrs","Value","AAV"]].isna().mean())
