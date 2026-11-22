from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 경로 설정
# =========================
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "outputs" / "paper" / "weight_grid" / "weight_grid_diagnostics.csv"
OUT_DIR = ROOT / "outputs" / "paper" / "weight_grid" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 데이터 불러오기
# =========================
df = pd.read_csv(CSV_PATH)

# 0.1 ~ 0.9만 사용
weights = np.round(np.arange(0.1, 1.0, 0.1), 1)
df = df[df["war_weight"].round(1).isin(weights)].copy()
df["war_weight"] = df["war_weight"].round(1)

role_order = ["hitter", "sp", "rp"]
role_labels = {
    "hitter": "Hitter",
    "sp": "Starting Pitcher",
    "rp": "Relief Pitcher",
}

# 검증
if len(df) != 27:
    raise ValueError(f"Expected 27 rows, found {len(df)}")

# =========================
# Figure 10 생성
# =========================
fig, axes = plt.subplots(
    nrows=3, ncols=2,
    figsize=(10, 10),
    sharex=True
)

metric_info = [
    ("accuracy", "Accuracy"),
    ("high_f1", "High-F1")
]

markers = {
    "hitter": "o",
    "sp": "s",
    "rp": "^"
}

for i, role in enumerate(role_order):
    sub = df[df["role"] == role].sort_values("war_weight")

    for j, (metric, ylabel) in enumerate(metric_info):
        ax = axes[i, j]

        ax.plot(
            sub["war_weight"],
            sub[metric],
            marker=markers[role],
            linewidth=2,
            markersize=5
        )

        # 기본 가중치 0.5 표시
        ax.axvline(
            0.5,
            linestyle="--",
            linewidth=1.2,
            alpha=0.7
        )

        # 패널 제목
        ax.set_title(f"{role_labels[role]} – {ylabel}")

        # 축 레이블
        if j == 0:
            ax.set_ylabel(ylabel)

        if i == 2:
            ax.set_xlabel("WAR Weight")

        # 눈금
        ax.set_xticks(weights)
        ax.grid(True, linestyle=":", alpha=0.35)

        # y축 범위
        if metric == "accuracy":
            ax.set_ylim(0.6, 1.0)
        else:
            ax.set_ylim(0.0, 0.75)

plt.tight_layout()

out_path = OUT_DIR / "Figure10_smallmultiples.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")