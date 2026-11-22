import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# CSV 파일 경로
csv_path = BASE_DIR / "outputs" / "final_3class" / "all_final_3class_metrics.csv"

print("현재 파일 위치:", Path(__file__).resolve())
print("프로젝트 루트:", BASE_DIR)
print("CSV 경로:", csv_path)
print("CSV 존재 여부:", csv_path.exists())

df = pd.read_csv(csv_path)

# 최종 기준만 선택
hitter = df[df["experiment"] == "hitter_final_3class_top40"].copy()
pitcher = df[df["experiment"] == "pitcher_final_3class_top20"].copy()

# 모델 순서 고정
model_order = ["logistic", "rf", "xgb"]
model_labels = ["Logistic Regression", "Random Forest", "XGBoost"]

hitter = hitter.set_index("model").loc[model_order].reset_index()
pitcher = pitcher.set_index("model").loc[model_order].reset_index()

# 공통 설정
x = np.arange(len(model_labels))
width = 0.35

# ===================
# Hitter Figure
# ===================
plt.figure(figsize=(7, 5))

plt.bar(x - width/2, hitter["accuracy"], width, label="Accuracy")
plt.bar(x + width/2, hitter["macro_f1"], width, label="Macro-F1")

plt.title("Performance Comparison of Final 3-class Models - Hitter")
plt.xticks(x, model_labels, rotation=15)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()


# ===================
# Pitcher Figure
# ===================
plt.figure(figsize=(7, 5))

plt.bar(x - width/2, pitcher["accuracy"], width, label="Accuracy")
plt.bar(x + width/2, pitcher["macro_f1"], width, label="Macro-F1")

plt.title("Performance Comparison of Final 3-class Models - Pitcher")
plt.xticks(x, model_labels, rotation=15)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()