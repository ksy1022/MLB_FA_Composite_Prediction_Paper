import pandas as pd

# 파일 읽기
df = pd.read_csv("data/results_threshold_final/all_experiment_metrics.csv")

# success_main만 사용 (2-class 핵심 결과)
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

# 그룹명 정리
def map_group(exp):
    if "hitter" in exp:
        return "Hitter"
    elif "pitcher_sp" in exp:
        return "Pitcher (SP)"
    elif "pitcher_rp" in exp:
        return "Pitcher (RP)"
    else:
        return "Other"

df["Group"] = df["experiment"].apply(map_group)

# feature set 정리
df["Features"] = df["experiment"].apply(
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

# 소수점 정리
metric_cols = [
    "accuracy", "roc_auc", "pr_auc",
    "recall_success", "f1_success"
]
df[metric_cols] = df[metric_cols].round(3)

# 논문용 표 구성
table4 = df[[
    "Group",
    "Model",
    "Features",
    "accuracy",
    "roc_auc",
    "pr_auc",
    "recall_success",
    "f1_success"
]].rename(columns={
    "accuracy": "Accuracy",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "recall_success": "Recall (Success)",
    "f1_success": "F1-score"
})

# 정렬 순서 지정
group_order = {
    "Hitter": 0,
    "Pitcher (SP)": 1,
    "Pitcher (RP)": 2
}
feature_order = {
    "Baseline": 0,
    "Expanded": 1
}

table4["group_order"] = table4["Group"].map(group_order)
table4["feature_order"] = table4["Features"].map(feature_order)

table4 = table4.sort_values(
    by=["group_order", "feature_order", "Model"]
).drop(columns=["group_order", "feature_order"])

print(table4.to_string(index=False))

# 엑셀 저장
table4.to_excel("Table4_2class_results.xlsx", index=False)
print("\nSaved: Table4_2class_results.xlsx")