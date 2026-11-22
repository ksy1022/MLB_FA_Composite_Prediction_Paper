from pathlib import Path
import time
import random
import requests
import pandas as pd
import numpy as np
from datetime import date

# =========================
# 프로젝트 경로
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_FA_DIR = PROJECT_ROOT / "data" / "interim" / "fa_contracts"

OUTPUT_PATH = INTERIM_FA_DIR / "mlb_fa_events_2007_2024.csv"

INTERIM_FA_DIR.mkdir(parents=True, exist_ok=True)


START_FA_YEAR = 2007
END_FA_YEAR = 2024

BASE = "https://statsapi.mlb.com/api/v1/transactions"

def fetch_transactions(start_date: str, end_date: str, limit: int = 2000) -> list[dict]:
    # Reddit에서 startDate/endDate 파라미터가 동작한다고 알려짐 :contentReference[oaicite:5]{index=5}
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "limit": limit,
        "order": "asc",
    }
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    # 보통 js["transactions"] 안에 리스트가 있음 (없으면 키를 출력해보면 됨)
    return js.get("transactions", [])

def to_fa_year(d: pd.Timestamp) -> int:
    # 오프시즌 거래(10~12월)는 다음 해 FA로 보는 규칙
    return d.year + 1 if d.month >= 10 else d.year

rows = []
for fa_year in range(START_FA_YEAR, END_FA_YEAR + 1):
    start = f"{fa_year-1}-10-01"
    end = f"{fa_year}-03-31"
    print(f"[INFO] FA {fa_year}: fetching {start} ~ {end}")

    txs = fetch_transactions(start, end, limit=5000)

    # 너무 잦은 호출 방지
    time.sleep(0.6 + random.uniform(0.2, 0.6))

    for t in txs:
        # 응답 구조는 약간씩 다를 수 있으니, 최대한 안전하게 꺼냄
        d = t.get("date") or t.get("effectiveDate") or ""
        desc = t.get("description", "")
        type_desc = ""
        if isinstance(t.get("typeDesc"), str):
            type_desc = t.get("typeDesc")
        elif isinstance(t.get("transactionType"), dict):
            type_desc = t["transactionType"].get("description", "")

        # 선수 id (MLBAM)
        person = t.get("person", {})
        player_mlbam = person.get("id")

        # 팀 정보
        from_team = None
        to_team = None
        if isinstance(t.get("fromTeam"), dict):
            from_team = t["fromTeam"].get("id")
        if isinstance(t.get("toTeam"), dict):
            to_team = t["toTeam"].get("id")

        # “FA 관련”만 남기기: description/typeDesc에 free agent가 들어가는 것 위주
        text = f"{type_desc} {desc}".lower()
        if "free agent" not in text:
            continue

        rows.append({
            "date": d,
            "fa_year": fa_year,                 # 우리가 루프 도는 FA년도(윈도우 anchor와 맞출 키)
            "fa_year_by_rule": None,            # (검증용) date 기반 규칙 fa_year
            "player_mlbam": player_mlbam,
            "type_desc": type_desc,
            "description": desc,
            "from_team": from_team,
            "to_team": to_team,
        })

df = pd.DataFrame(rows)

# 날짜 파싱 + 규칙 기반 fa_year 추가(검증용)
df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
df["fa_year_by_rule"] = df["date_parsed"].apply(lambda x: to_fa_year(x) if pd.notna(x) else np.nan)

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(f"[OK] Saved: {OUTPUT_PATH}")
print(df[["fa_year","type_desc"]].value_counts().head(20))
