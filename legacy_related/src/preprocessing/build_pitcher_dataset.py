# make_final_pitcher_dataset.py
from pathlib import Path
import pandas as pd
import numpy as np
import re
import unicodedata
import matplotlib.pyplot as plt
import matplotlib

# ---- (선택) 한글 폰트 경고 방지: Windows ----
matplotlib.rcParams["font.family"] = "Malgun Gothic"   # 맑은 고딕
matplotlib.rcParams["axes.unicode_minus"] = False

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
    """
    s = str(s)
    s = strip_accents(s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)      # 문자/숫자/공백/하이픈만 남김
    s = re.sub(r"\s+", " ", s).strip()  # 공백 정리
    return s

# =========================
# 2) 로드
# =========================
# =========================
# 2) 경로 설정 및 로드
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_FA_DIR = PROJECT_ROOT / "data" / "interim" /"player_windows"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PIT_PATH = INTERIM_FA_DIR / "mlb_pitcher_fa_windows.csv"
SPOTRAC_PATH = PROCESSED_DIR / "spotrac_fa_2011_2024_clean.csv"

pit = pd.read_csv(PIT_PATH)
fa = pd.read_csv(SPOTRAC_PATH)

# 컬럼명 공백 방지
pit.columns = pit.columns.astype(str).str.strip()
fa.columns  = fa.columns.astype(str).str.strip()

# =========================
# 3) 기간 필터
# =========================
pit = pit[(pit["anchor_year"] >= 2011) & (pit["anchor_year"] <= 2024)].copy()
fa  = fa[(fa["fa_year"] >= 2011) & (fa["fa_year"] <= 2024)].copy()

# =========================
# 4) 이름 정규화 키
# =========================
# pitcher windows 쪽 선수 이름 컬럼은 보통 'player'
if "player" not in pit.columns:
    raise RuntimeError(f"'player' 컬럼이 없습니다. 현재 컬럼 예시: {pit.columns.tolist()[:30]}")

pit["player_norm"] = pit["player"].apply(normalize_name)
fa["player_norm"]  = fa["Player"].apply(normalize_name)

# =========================
# 5) Spotrac 중복 제거 (같은 선수/연도면 Value 큰 계약 1개만)
# =========================
fa_small = (
    fa[["player_norm", "fa_year", "Pos", "Yrs", "Value", "AAV"]]
    .sort_values(["player_norm", "fa_year", "Value"], ascending=[True, True, False])
    .drop_duplicates(["player_norm", "fa_year"], keep="first")
    .copy()
)

# =========================
# 6) Merge: (player_norm, anchor_year) == (player_norm, fa_year)
# =========================
merged = pit.merge(
    fa_small,
    left_on=["player_norm", "anchor_year"],
    right_on=["player_norm", "fa_year"],
    how="left"
)

# =========================
# 7) 라벨 생성
# =========================
merged["is_real_fa"] = merged["fa_year"].notna().astype(int)

# =========================
# 8) sanity check 출력
# =========================
match_rate = merged["is_real_fa"].mean()
print(f"[INFO] rows (pitcher windows, 2011-2024): {len(pit):,}")
print(f"[INFO] rows (merged): {len(merged):,}")
print(f"[INFO] match rate (is_real_fa=1): {match_rate:.4f}")
print(f"[INFO] FA count: {merged['is_real_fa'].sum():,}")

# =========================
# 9) 저장
# =========================
OUT_ALL = PROCESSED_DIR / "final_pitcher_fa_dataset_2011_2024.csv"
OUT_FA = PROCESSED_DIR / "final_pitcher_fa_dataset_2011_2024_FAonly.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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
# 10) 간단 그래프
# =========================
counts = merged["is_real_fa"].value_counts().sort_index()
plt.figure()
plt.bar(["비FA(0)", "FA(1)"], [counts.get(0, 0), counts.get(1, 0)])
plt.title("투수: 실제 FA 계약 여부 분포")
plt.xlabel("구분")
plt.ylabel("표본 수")
plt.tight_layout()
plt.show()

fa_only = merged[merged["is_real_fa"] == 1].copy()
if len(fa_only) > 0:
    fa_only["log_value"] = np.log1p(fa_only["Value"])
    fa_only["log_aav"]   = np.log1p(fa_only["AAV"])

    plt.figure()
    plt.hist(fa_only["log_value"].dropna(), bins=30)
    plt.title("투수 FA 계약 총액(Value) 분포 (log1p)")
    plt.xlabel("log1p(Value)")
    plt.ylabel("빈도")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(fa_only["log_aav"].dropna(), bins=30)
    plt.title("투수 FA 연평균(AAV) 분포 (log1p)")
    plt.xlabel("log1p(AAV)")
    plt.ylabel("빈도")
    plt.tight_layout()
    plt.show()

# =========================
# 11) 요약 통계 (Table 1 느낌)
# =========================
num_cols = [
    "age_at_contract",
    "pre_IP_3y",
    "pre_WAR_3y_sum",
    "pre_WAR_3y_avg",
    "pre_WAR_3y_std",
    "pre_ERA_3y_avg",
    "pre_FIP_3y_avg",
    "pre_KBB_3y_avg",
    "pre_GS_3y_sum",
    "pre_G_3y_sum",
    "post_WAR_3y_sum",
    "post_IP_3y",
]
num_cols = [c for c in num_cols if c in merged.columns]

summary = merged.groupby("is_real_fa")[num_cols].agg(["count", "mean", "std", "median"])
print("\n[Table 1] 투수 FA(1) vs 비FA(0) 요약 통계")
print(summary)
