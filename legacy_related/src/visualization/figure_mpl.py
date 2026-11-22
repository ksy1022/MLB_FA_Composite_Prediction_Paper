import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 파일 경로
# =========================
HITTER_DATA = "data/final_hitter_fa_dataset_2011_2024_FAonly_clean_posgroup.csv"
PITCHER_DATA = "data/final_pitcher_fa_dataset_2011_2024_FAonly_clean.csv"
METRICS_3CLASS = "data/results_final_3class/all_final_3class_metrics.csv"
METRICS_2CLASS = "data/results_threshold_final/all_experiment_metrics.csv"
OUTDIR = "figures"

os.makedirs(OUTDIR, exist_ok=True)

# =========================
# 공통 함수
# =========================
def zscore_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    s = s.fillna(s.median())
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std

def prettify_model_name(x: str) -> str:
    mapping = {
        "logistic": "Logistic Regression",
        "rf": "Random Forest",
        "xgb": "XGBoost",
        "stack": "Stacking Ensemble"
    }
    return mapping.get(x, x)

# =========================
# 데이터 불러오기
# =========================
hitter = pd.read_csv(HITTER_DATA)
pitcher = pd.read_csv(PITCHER_DATA)

metrics3 = pd.read_csv(METRICS_3CLASS)
metrics2 = pd.read_csv(METRICS_2CLASS)

# =========================================================
# Figure 2. Composite score 분포
# =========================================================
# hitter: post_WAR_3y_sum + post_PA_3y
# pitcher: post_WAR_3y_sum + post_IP_3y
# =========================================================

hitter["z_post_WAR"] = zscore_series(hitter["post_WAR_3y_sum"])
hitter["z_post_playtime"] = zscore_series(hitter["post_PA_3y"])
hitter["composite_score"] = hitter["z_post_WAR"] + hitter["z_post_playtime"]

pitcher["z_post_WAR"] = zscore_series(pitcher["post_WAR_3y_sum"])
pitcher["z_post_playtime"] = zscore_series(pitcher["post_IP_3y"])
pitcher["composite_score"] = pitcher["z_post_WAR"] + pitcher["z_post_playtime"]

plt.figure(figsize=(8, 5))
plt.hist(hitter["composite_score"].dropna(), bins=30, alpha=0.6, label="Hitter")
plt.hist(pitcher["composite_score"].dropna(), bins=30, alpha=0.6, label="Pitcher")
plt.xlabel("Composite Score")
plt.ylabel("Frequency")
plt.title("Figure 2. Distribution of Composite Performance Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure2_composite_distribution.png"), dpi=300)
plt.close()

# =========================================================
# Figure 3. Percentile 기준 시각화
# =========================================================
# hitter: top 40%
# pitcher: top 20%
# plus 20/30/40 lines 예시
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Hitter
h_vals = hitter["composite_score"].dropna()
h_cuts = {
    "Top 20%": h_vals.quantile(0.80),
    "Top 30%": h_vals.quantile(0.70),
    "Top 40%": h_vals.quantile(0.60),
}
axes[0].hist(h_vals, bins=30, alpha=0.7)
for label, cut in h_cuts.items():
    axes[0].axvline(cut, linestyle="--", linewidth=1.5, label=f"{label}: {cut:.2f}")
axes[0].set_title("Hitter")
axes[0].set_xlabel("Composite Score")
axes[0].set_ylabel("Frequency")
axes[0].legend(fontsize=8)

# Pitcher
p_vals = pitcher["composite_score"].dropna()
p_cuts = {
    "Top 20%": p_vals.quantile(0.80),
    "Top 30%": p_vals.quantile(0.70),
    "Top 40%": p_vals.quantile(0.60),
}
axes[1].hist(p_vals, bins=30, alpha=0.7)
for label, cut in p_cuts.items():
    axes[1].axvline(cut, linestyle="--", linewidth=1.5, label=f"{label}: {cut:.2f}")
axes[1].set_title("Pitcher")
axes[1].set_xlabel("Composite Score")
axes[1].set_ylabel("Frequency")
axes[1].legend(fontsize=8)

fig.suptitle("Figure 3. Percentile Thresholds for Success Classification", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure3_percentile_thresholds.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# Figure 4. 3-class 성능 비교
# =========================================================
# accuracy / macro_f1 vs success_top_ratio
# hitter / pitcher 각각 하나의 figure로 만들어도 되지만
# 여기서는 한 그림에 2개 subplot
# =========================================================

metrics3 = metrics3.copy()
metrics3["Group"] = metrics3["experiment"].apply(lambda x: "Hitter" if "hitter" in x.lower() else "Pitcher")
metrics3["Model"] = metrics3["model"].apply(prettify_model_name)
metrics3["Top %"] = (metrics3["success_top_ratio"] * 100).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

for model_name in metrics3["Model"].unique():
    sub_h = metrics3[(metrics3["Group"] == "Hitter") & (metrics3["Model"] == model_name)].sort_values("Top %")
    if not sub_h.empty:
        axes[0].plot(sub_h["Top %"], sub_h["accuracy"], marker="o", label=f"{model_name} (Acc)")
        axes[0].plot(sub_h["Top %"], sub_h["macro_f1"], marker="s", linestyle="--", label=f"{model_name} (Macro-F1)")

    sub_p = metrics3[(metrics3["Group"] == "Pitcher") & (metrics3["Model"] == model_name)].sort_values("Top %")
    if not sub_p.empty:
        axes[1].plot(sub_p["Top %"], sub_p["accuracy"], marker="o", label=f"{model_name} (Acc)")
        axes[1].plot(sub_p["Top %"], sub_p["macro_f1"], marker="s", linestyle="--", label=f"{model_name} (Macro-F1)")

axes[0].set_title("Hitter")
axes[1].set_title("Pitcher")

for ax in axes:
    ax.set_xlabel("Success Percentile Threshold (%)")
    ax.set_ylabel("Score")
    ax.set_xticks([20, 30, 40])
    ax.legend(fontsize=7)

fig.suptitle("Figure 4. Performance Comparison across Percentile Thresholds (3-class)", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure4_3class_performance.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# Figure 5. RP에서 accuracy vs recall 비교
# =========================================================
# baseline / expanded 나눠서 모델별 비교
# =========================================================

rp = metrics2[metrics2["experiment"].isin(["pitcher_rp_baseline_main", "pitcher_rp_expanded_main"])].copy()
rp["Feature Set"] = rp["experiment"].apply(lambda x: "Baseline" if "baseline" in x else "Expanded")
rp["Model"] = rp["model"].apply(prettify_model_name)

# 보기 좋게 정렬
model_order = ["Logistic Regression", "Random Forest", "XGBoost", "Stacking Ensemble"]
rp["Model"] = pd.Categorical(rp["Model"], categories=model_order, ordered=True)
rp = rp.sort_values(["Feature Set", "Model"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

for i, feature_set in enumerate(["Baseline", "Expanded"]):
    sub = rp[rp["Feature Set"] == feature_set].copy()
    x = np.arange(len(sub))
    width = 0.35

    axes[i].bar(x - width/2, sub["accuracy"], width, label="Accuracy")
    axes[i].bar(x + width/2, sub["recall_success"], width, label="Recall")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(sub["Model"], rotation=20, ha="right")
    axes[i].set_ylim(0, 1.05)
    axes[i].set_title(f"RP - {feature_set}")
    axes[i].set_ylabel("Score")
    axes[i].legend()

fig.suptitle("Figure 5. Accuracy and Recall in RP Models", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure5_rp_accuracy_recall.png"), dpi=300, bbox_inches="tight")
plt.close()

print("모든 figure 저장 완료")
print(f"저장 폴더: {OUTDIR}")