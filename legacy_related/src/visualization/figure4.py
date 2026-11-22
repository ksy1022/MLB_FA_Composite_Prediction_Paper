import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/results_final_3class/all_final_3class_metrics.csv")

# 그룹 구분
df["Group"] = df["experiment"].apply(
    lambda x: "Hitter" if "hitter" in x.lower() else "Pitcher"
)

df["Top %"] = (df["success_top_ratio"] * 100).astype(int)

# -------------------------
# 최종 기준 선택
# -------------------------
hitter_df = df[(df["Group"] == "Hitter") & (df["Top %"] == 40)].copy()
pitcher_df = df[(df["Group"] == "Pitcher") & (df["Top %"] == 20)].copy()

# 모델 정렬
hitter_df["model_order"] = hitter_df["model"].str.lower()
pitcher_df["model_order"] = pitcher_df["model"].str.lower()

hitter_df = hitter_df.sort_values("model_order")
pitcher_df = pitcher_df.sort_values("model_order")

models = hitter_df["model"].values

# 값 추출
hitter_acc = hitter_df["accuracy"].values
hitter_f1  = hitter_df["macro_f1"].values

pitcher_acc = pitcher_df["accuracy"].values
pitcher_f1  = pitcher_df["macro_f1"].values

# -------------------------
# 그래프
# -------------------------
x = np.arange(len(models))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# ---------- Hitter ----------
bars1 = axes[0].bar(x - width/2, hitter_acc, width, label="Accuracy")
bars2 = axes[0].bar(x + width/2, hitter_f1, width, label="Macro-F1")

# 수치 표시
for i, v in enumerate(hitter_acc):
    axes[0].text(i - width/2, v + 0.015, f"{v:.2f}", ha='center', fontsize=8)
for i, v in enumerate(hitter_f1):
    axes[0].text(i + width/2, v + 0.015, f"{v:.2f}", ha='center', fontsize=8)

axes[0].axhline(1/3, linestyle="--", linewidth=1, label="Baseline (0.33)")
axes[0].set_title("Hitter (Top 40%)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=15)
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 0.8)  # 🔥 여유 확보

# ---------- Pitcher ----------
bars3 = axes[1].bar(x - width/2, pitcher_acc, width, label="Accuracy")
bars4 = axes[1].bar(x + width/2, pitcher_f1, width, label="Macro-F1")

for i, v in enumerate(pitcher_acc):
    axes[1].text(i - width/2, v + 0.015, f"{v:.2f}", ha='center', fontsize=8)
for i, v in enumerate(pitcher_f1):
    axes[1].text(i + width/2, v + 0.015, f"{v:.2f}", ha='center', fontsize=8)

axes[1].axhline(1/3, linestyle="--", linewidth=1, label="Baseline (0.33)")
axes[1].set_title("Pitcher (Top 20%)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=15)
axes[1].set_ylim(0, 0.8)  # 🔥 여유 확보

# -------------------------
# 공통 legend
# -------------------------
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),  # 1.05 → 0.98로 수정
    ncol=3,
    frameon=True
)

plt.tight_layout(rect=[0, 0, 1, 0.90])  # 위쪽 legend 공간 확보
plt.savefig("figure3_final_performance.png", dpi=300, bbox_inches="tight")
plt.show()