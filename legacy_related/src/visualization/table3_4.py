import pandas as pd

# 파일 읽기
df = pd.read_csv("data/results_final_3class/all_final_3class_metrics.csv")

# 보기 좋게 모델명 바꾸기
model_map = {
    "logistic": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost"
}
df["Model"] = df["model"].map(model_map)

# 타자 / 투수 구분
df["Group"] = df["experiment"].apply(
    lambda x: "Hitter" if "hitter" in x.lower() else "Pitcher"
)

# success criterion 보기 좋게
df["Success Criterion"] = df["success_top_ratio"].apply(
    lambda x: f"Top {int(x*100)}%"
)

# 소수점 정리
for col in ["accuracy", "macro_f1", "weighted_f1", "success_cut"]:
    df[col] = df[col].round(4)

# 타자 표
table_hitter = df[df["Group"] == "Hitter"][
    ["Success Criterion", "Model", "accuracy", "macro_f1", "weighted_f1"]
].rename(columns={
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1"
}).sort_values(["Success Criterion", "Model"])

# 투수 표
table_pitcher = df[df["Group"] == "Pitcher"][
    ["Success Criterion", "Model", "accuracy", "macro_f1", "weighted_f1"]
].rename(columns={
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1"
}).sort_values(["Success Criterion", "Model"])

print("=== Hitter 3-class Table ===")
print(table_hitter.to_string(index=False))

print("\n=== Pitcher 3-class Table ===")
print(table_pitcher.to_string(index=False))

# 엑셀로 저장
with pd.ExcelWriter("table3_3class_results.xlsx") as writer:
    table_hitter.to_excel(writer, sheet_name="Hitter_3class", index=False)
    table_pitcher.to_excel(writer, sheet_name="Pitcher_3class", index=False)