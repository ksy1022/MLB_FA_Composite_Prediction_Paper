import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 파일 읽기
df = pd.read_csv("data/results_threshold_final/all_experiment_metrics.csv")

# 타자 관련 실험만 선택
hitter_df = df[df["experiment"].isin([
    "hitter_baseline_main",
    "hitter_expanded_main"
])].copy()

# feature set 구분
hitter_df["Feature_Set"] = hitter_df["experiment"].apply(
    lambda x: "Baseline" if "baseline" in x else "Expanded"
)

# 모델명 정리
model_map = {
    "logistic": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "stack": "Stacking Ensemble"
}
hitter_df["Model"] = hitter_df["model"].map(model_map)

# pivot
pivot_h = hitter_df.pivot_table(
    index="Model",
    columns="Feature_Set",
    values=["accuracy", "recall_success"]
)

pivot_h.columns = [f"{metric}_{feature}".lower() for metric, feature in pivot_h.columns]
pivot_h = pivot_h.reset_index()

# 모델 순서 정렬
model_order = ["Logistic Regression", "Random Forest", "XGBoost", "Stacking Ensemble"]
pivot_h["Model"] = pd.Categorical(pivot_h["Model"], categories=model_order, ordered=True)
pivot_h = pivot_h.sort_values("Model")

# Figure
x = np.arange(len(pivot_h))
width = 0.18

plt.figure(figsize=(10, 5))

plt.bar(x - 1.5*width, pivot_h["accuracy_baseline"], width, label="Baseline Accuracy")
plt.bar(x - 0.5*width, pivot_h["accuracy_expanded"], width, label="Expanded Accuracy")
plt.bar(x + 0.5*width, pivot_h["recall_success_baseline"], width, label="Baseline Recall")
plt.bar(x + 1.5*width, pivot_h["recall_success_expanded"], width, label="Expanded Recall")

plt.xticks(x, pivot_h["Model"], rotation=20, ha="right")
plt.ylim(0, 1.0)
plt.ylabel("Score")
# plt.title("Hitter: Baseline vs Expanded Performance")
plt.grid(axis="y", alpha=0.3)
plt.legend(ncol=2)

plt.tight_layout()
plt.savefig("figure5_hitter_baseline_expanded.png", dpi=300, bbox_inches="tight")
plt.show()