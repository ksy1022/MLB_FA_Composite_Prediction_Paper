# make_final_hitter_dataset.py
from pathlib import Path
import pandas as pd
import numpy as np
import re
import unicodedata
import matplotlib.pyplot as plt
import matplotlib


# ---- 한글 폰트 설정 (Windows) ----
matplotlib.rcParams["font.family"] = "Malgun Gothic"   # 맑은 고딕
matplotlib.rcParams["axes.unicode_minus"] = False      # 마이너스 깨짐 방지

# =========================
# 1) 유틸: 이름 정규화
# =========================
def strip_accents(s: str) -> str:
    """Adrián -> Adrian"""
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def normalize_name(s: str) -> str:
    """
    - 소문자
    - 악센트 제거
    - 특수문자 제거
    - 공백 정리
    - Jr/Sr/III 등은 일단 남겨도 되지만, 매칭률 낮으면 추가 처리 가능
    """
    s = str(s)
    s = strip_accents(s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)     # 문자/숫자/공백/하이픈만 남김
    s = re.sub(r"\s+", " ", s).strip() # 공백 정리
    return s

# =========================
# 2) 로드
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_FA_DIR = PROJECT_ROOT / "data" / "interim" /"player_windows"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HIT_PATH = INTERIM_FA_DIR / "mlb_hitter_fa_windows.csv"

SPOTRAC_PATH = (
    PROCESSED_DIR / "spotrac_fa_2011_2024_clean.csv"
)

hit = pd.read_csv(HIT_PATH)
fa = pd.read_csv(SPOTRAC_PATH)

# 컬럼명 공백 방지
hit.columns = hit.columns.astype(str).str.strip()
fa.columns = fa.columns.astype(str).str.strip()

# =========================
# 3) 기간 필터 (Spotrac이 2011~2024니까)
# =========================
hit = hit[(hit["anchor_year"] >= 2011) & (hit["anchor_year"] <= 2024)].copy()
fa = fa[(fa["fa_year"] >= 2011) & (fa["fa_year"] <= 2024)].copy()

# =========================
# 4) 이름 정규화 키 만들기
# =========================
hit["player_norm"] = hit["player"].apply(normalize_name)
fa["player_norm"] = fa["Player"].apply(normalize_name)

# =========================
# 5) (핵심) Merge: anchor_year == fa_year AND player_norm 일치
# =========================
# 같은 (player_norm, fa_year) 중복이면 Value 큰 계약 1개만 남김
fa_small = (
    fa[["player_norm", "fa_year", "Pos", "Yrs", "Value", "AAV"]]
    .sort_values(["player_norm", "fa_year", "Value"], ascending=[True, True, False])
    .drop_duplicates(["player_norm", "fa_year"], keep="first")
    .copy()
)


merged = hit.merge(
    fa_small,
    left_on=["player_norm", "anchor_year"],
    right_on=["player_norm", "fa_year"],
    how="left"
)

# =========================
# 6) 라벨 생성
# =========================
merged["is_real_fa"] = merged["fa_year"].notna().astype(int)

# =========================
# 7) sanity check 출력
# =========================
match_rate = merged["is_real_fa"].mean()
print(f"[INFO] rows (hit windows, 2011-2024): {len(hit):,}")
print(f"[INFO] rows (merged): {len(merged):,}")
print(f"[INFO] match rate (is_real_fa=1): {match_rate:.4f}")

# =========================
# 저장 경로
# =========================
OUT_ALL = (
    PROCESSED_DIR / "final_hitter_fa_dataset_2011_2024.csv"
)

OUT_FA = (
    PROCESSED_DIR / "final_hitter_fa_dataset_2011_2024_FAonly.csv"
)

# processed 폴더 자동 생성
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 저장
merged.to_csv(
    OUT_ALL,
    index=False,
    encoding="utf-8-sig"
)

merged[merged["is_real_fa"] == 1].to_csv(
    OUT_FA,
    index=False,
    encoding="utf-8-sig"
)

print(f"[OK] Saved: {OUT_ALL}")
print(f"[OK] Saved: {OUT_FA}")

# =========================
# 9) (그래프1) 실제 FA vs 비FA 개수
# =========================
counts = merged["is_real_fa"].value_counts().sort_index()
plt.figure()
plt.bar(["비FA(0)", "FA(1)"], [counts.get(0,0), counts.get(1,0)])
plt.title("실제 FA 계약 여부 분포")
plt.xlabel("구분")
plt.ylabel("표본 수")
plt.tight_layout()
plt.show()

# =========================
# 10) (그래프2) FA만 계약금액 분포 (Value / AAV)
# =========================
fa_only = merged[merged["is_real_fa"] == 1].copy()

if len(fa_only) > 0:
    # 로그 스케일 추천(분포가 치우쳐 있음)
    fa_only["log_value"] = np.log1p(fa_only["Value"])
    fa_only["log_aav"] = np.log1p(fa_only["AAV"])

    plt.figure()
    plt.hist(fa_only["log_value"].dropna(), bins=30)
    plt.title("FA 계약 총액(Value) 분포 (log1p)")
    plt.xlabel("log1p(Value)")
    plt.ylabel("빈도")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(fa_only["log_aav"].dropna(), bins=30)
    plt.title("FA 연평균(AAV) 분포 (log1p)")
    plt.xlabel("log1p(AAV)")
    plt.ylabel("빈도")
    plt.tight_layout()
    plt.show()

# =========================
# 11) Table 1 스타일 요약 (FA vs 비FA)
# =========================
num_cols = [
    "age_at_contract",
    "pre_PA_3y",
    "pre_WAR_3y_sum",
    "pre_WAR_3y_avg",
    "pre_WAR_3y_std",
    "pre_OPSplus_3y_avg",
    "pre_BB_rate_3y_avg",
    "pre_K_rate_3y_avg",
    "post_WAR_3y_sum",
    "post_PA_3y",
]

# 존재하는 컬럼만
num_cols = [c for c in num_cols if c in merged.columns]

summary = (
    merged
    .groupby("is_real_fa")[num_cols]
    .agg(["count", "mean", "std", "median"])
)

print("\n[Table 1] FA(1) vs 비FA(0) 요약 통계")
print(summary)
