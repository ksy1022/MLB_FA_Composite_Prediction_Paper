import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 파일 읽기
df = pd.read_csv("data/results_threshold_final/all_experiment_metrics.csv")

# SP / RP 관련 실험만 선택
pitcher_df = df[df["experiment"].isin([
    "pitcher_sp_baseline_main",
    "pitcher_sp_expanded_main",
    "pitcher_rp_baseline_main",
    "pitcher_rp_expanded_main"
])].copy()

# 그룹 구분
def map_group(exp):
    if "pitcher_sp" in exp:
        return "Pitcher (SP)"
    elif "pitcher_rp" in exp:
        return "Pitcher (RP)"
    return "Other"

pitcher_df["Group"] = pitcher_df["experiment"].apply(map_group)

# feature set 구분
pitcher_df["Feature_Set"] = pitcher_df["experiment"].apply(
    lambda x: "Baseline" if "baseline" in x else "Expanded"
)

# 모델명 정리
model_map = {
    "logistic": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "stack": "Stacking Ensemble"
}
pitcher_df["Model"] = pitcher_df["model"].map(model_map)

# 모델 순서
model_order = ["Logistic Regression", "Random Forest", "XGBoost", "Stacking Ensemble"]

# Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, group in zip(axes, ["Pitcher (SP)", "Pitcher (RP)"]):
    sub = pitcher_df[pitcher_df["Group"] == group].copy()

    pivot = sub.pivot_table(
        index="Model",
        columns="Feature_Set",
        values=["accuracy", "recall_success"]
    )

    pivot.columns = [f"{metric}_{feature}".lower() for metric, feature in pivot.columns]
    pivot = pivot.reset_index()

    pivot["Model"] = pd.Categorical(pivot["Model"], categories=model_order, ordered=True)
    pivot = pivot.sort_values("Model")

    x = np.arange(len(pivot))
    width = 0.18

    ax.bar(x - 1.5*width, pivot["accuracy_baseline"], width, label="Baseline Accuracy")
    ax.bar(x - 0.5*width, pivot["accuracy_expanded"], width, label="Expanded Accuracy")
    ax.bar(x + 0.5*width, pivot["recall_success_baseline"], width, label="Baseline Recall")
    ax.bar(x + 1.5*width, pivot["recall_success_expanded"], width, label="Expanded Recall")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title(group)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))

# plt.suptitle("Pitcher: Baseline vs Expanded Performance by Role", y=1.12)
plt.tight_layout()
plt.savefig("figure6_pitcher_sp_rp_comparison.png", dpi=300, bbox_inches="tight")
plt.show()