from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FA_DIR = PROJECT_ROOT / "data" / "interim" / "fa_contracts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = RAW_FA_DIR / "spotrac_fa_2011_2024.csv"

OUTPUT_PATH = (
    PROCESSED_DIR / "spotrac_fa_2011_2024_clean.csv"
)

fa = pd.read_csv(INPUT_PATH)

# 1) 기본 정리: Yrs, Value가 0인 행 제거
fa_clean = fa[(fa["Yrs"] >= 1) & (fa["Value"] > 0)].copy()

# 2) AAV가 비어있으면 재계산
fa_clean["AAV"] = fa_clean["AAV"].fillna(fa_clean["Value"] / fa_clean["Yrs"])

# 3) 확인
print("Before:", fa.shape)
print("After :", fa_clean.shape)
print("AAV NaN rate:", fa_clean["AAV"].isna().mean())

# 4) 저장
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

fa_clean.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(f"[OK] Saved: {OUTPUT_PATH}")