import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


# CSV 파일 경로
csv_path = BASE_DIR / "outputs" / "final_2class" / "all_experiment_metrics.csv"

print("현재 파일 위치:", Path(__file__).resolve())
print("프로젝트 루트:", BASE_DIR)
print("CSV 경로:", csv_path)
print("CSV 존재 여부:", csv_path.exists())

df = pd.read_csv(csv_path)

# 타자 데이터
hitter = df[df["experiment"].isin(["hitter_baseline_main", "hitter_expanded_main"])].copy()

hitter["feature_set"] = hitter["experiment"].apply(
    lambda x: "Baseline" if "baseline" in x else "Expanded"
)

model_order = ["logistic", "rf", "xgb"]
labels = ["LR", "RF", "XGB"]

baseline = hitter[hitter["feature_set"] == "Baseline"].set_index("model").loc[model_order]
expanded = hitter[hitter["feature_set"] == "Expanded"].set_index("model").loc[model_order]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(7,5))

plt.bar(x - width, baseline["accuracy"], width, label="Baseline Acc")
plt.bar(x, expanded["accuracy"], width, label="Expanded Acc")
plt.bar(x + width, expanded["recall_success"], width, label="Expanded Recall")

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Performance Comparison in 2-class Classification (Hitter)")
plt.legend()
plt.tight_layout()

plt.savefig("figure4_hitter.png", dpi=300)
plt.show()

# SP + RP 합쳐서 그림

pitcher = df[df["experiment"].isin([
    "pitcher_sp_expanded_main",
    "pitcher_rp_expanded_main"
])].copy()

pitcher["type"] = pitcher["experiment"].apply(
    lambda x: "SP" if "sp" in x else "RP"
)

sp = pitcher[pitcher["type"] == "SP"].set_index("model").loc[model_order]
rp = pitcher[pitcher["type"] == "RP"].set_index("model").loc[model_order]

plt.figure(figsize=(8, 5))

plt.plot(labels, sp["accuracy"], marker="o", label="SP Accuracy")
plt.plot(labels, sp["recall_success"], marker="o", linestyle="--", label="SP Recall")
plt.plot(labels, rp["accuracy"], marker="s", label="RP Accuracy")
plt.plot(labels, rp["recall_success"], marker="s", linestyle="--", label="RP Recall")

plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("Performance Comparison in 2-class Classification (Pitcher)")
plt.legend()
plt.tight_layout()
plt.savefig("figure5_pitcher_line.png", dpi=300)
plt.show()