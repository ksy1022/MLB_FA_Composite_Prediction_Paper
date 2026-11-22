import pandas as pd

# 파일 읽기
df = pd.read_csv("data/results_threshold_final/all_experiment_metrics.csv")

# 2-class 핵심 결과만 사용
df = df[df["target"].isin(["h_success_main", "p_success_main"])].copy()

# 필요한 실험만 선택
wanted_experiments = [
    "hitter_baseline_main",
    "hitter_expanded_main",
    "pitcher_sp_baseline_main",
    "pitcher_sp_expanded_main",
    "pitcher_rp_baseline_main",
    "pitcher_rp_expanded_main",
]
df = df[df["experiment"].isin(wanted_experiments)].copy()

# Group 지정
def map_group(exp):
    if "hitter" in exp:
        return "Hitter"
    elif "pitcher_sp" in exp:
        return "Pitcher (SP)"
    elif "pitcher_rp" in exp:
        return "Pitcher (RP)"
    return "Other"

df["Group"] = df["experiment"].apply(map_group)

# Feature set 지정
df["Feature_Set"] = df["experiment"].apply(
    lambda x: "Baseline" if "baseline" in x else "Expanded"
)

# 모델명 정리
model_map = {
    "logistic": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "stack": "Stacking Ensemble"
}
df["Model"] = df["model"].map(model_map)

# 필요한 컬럼만 선택
df = df[[
    "Group", "Model", "Feature_Set",
    "accuracy", "recall_success", "pr_auc"
]].copy()

# pivot으로 baseline / expanded 나누기
pivot = df.pivot_table(
    index=["Group", "Model"],
    columns="Feature_Set",
    values=["accuracy", "recall_success", "pr_auc"]
)

# 컬럼 평탄화
pivot.columns = [
    f"{metric}_{feature}".lower()
    for metric, feature in pivot.columns
]
pivot = pivot.reset_index()

# 차이 계산
pivot["delta_accuracy"] = pivot["accuracy_expanded"] - pivot["accuracy_baseline"]
pivot["delta_recall"] = pivot["recall_success_expanded"] - pivot["recall_success_baseline"]

# 컬럼 이름 논문 스타일로 변경
table4 = pivot.rename(columns={
    "accuracy_baseline": "Baseline Accuracy",
    "accuracy_expanded": "Expanded Accuracy",
    "delta_accuracy": "Δ Accuracy",
    "recall_success_baseline": "Baseline Recall",
    "recall_success_expanded": "Expanded Recall",
    "delta_recall": "Δ Recall",
    "pr_auc_baseline": "Baseline PR-AUC",
    "pr_auc_expanded": "Expanded PR-AUC"
})

# 소수점 정리
num_cols = [
    "Baseline Accuracy", "Expanded Accuracy", "Δ Accuracy",
    "Baseline Recall", "Expanded Recall", "Δ Recall",
    "Baseline PR-AUC", "Expanded PR-AUC"
]
table4[num_cols] = table4[num_cols].round(3)

# 보기 좋은 정렬
group_order = {
    "Hitter": 0,
    "Pitcher (SP)": 1,
    "Pitcher (RP)": 2
}
table4["group_order"] = table4["Group"].map(group_order)
table4 = table4.sort_values(["group_order", "Model"]).drop(columns=["group_order"])

print(table4.to_string(index=False))

# 저장
table4.to_excel("Table4_same_model_comparison.xlsx", index=False)
print("\nSaved: Table4_same_model_comparison.xlsx")