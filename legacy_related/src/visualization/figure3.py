import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
hitter = pd.read_csv("data/final_hitter_fa_dataset_2011_2024_FAonly_clean_posgroup.csv")
pitcher = pd.read_csv("data/final_pitcher_fa_dataset_2011_2024_FAonly_clean.csv")

# z-score 함수
def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.fillna(s.median())
    return (s - s.mean()) / s.std()

# composite score 생성
hitter["z_WAR"] = zscore(hitter["post_WAR_3y_sum"])
hitter["z_playtime"] = zscore(hitter["post_PA_3y"])
hitter["composite_score"] = hitter["z_WAR"] + hitter["z_playtime"]

pitcher["z_WAR"] = zscore(pitcher["post_WAR_3y_sum"])
pitcher["z_playtime"] = zscore(pitcher["post_IP_3y"])
pitcher["composite_score"] = pitcher["z_WAR"] + pitcher["z_playtime"]

# -------------------------
# Hitter figure
# -------------------------
h_vals = hitter["composite_score"].dropna()

h_cut_20 = h_vals.quantile(0.80)  # top 20%
h_cut_30 = h_vals.quantile(0.70)  # top 30%
h_cut_40 = h_vals.quantile(0.60)  # top 40% (selected)

plt.figure(figsize=(7, 5))
plt.hist(h_vals, bins=30, alpha=0.7)

plt.axvline(h_cut_20, linestyle="--", linewidth=1.5, label=f"Top 20% ({h_cut_20:.2f})")
plt.axvline(h_cut_30, linestyle="--", linewidth=1.5, label=f"Top 30% ({h_cut_30:.2f})")
plt.axvline(h_cut_40, linestyle="-", linewidth=2.5, label=f"Top 40% ({h_cut_40:.2f}) [Selected]")

# plt.title("Hitter: Percentile Thresholds for Success Classification")
plt.xlabel("Composite Score")
plt.ylabel("Frequency")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figure3a_hitter_percentile_thresholds.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------
# Pitcher figure
# -------------------------
p_vals = pitcher["composite_score"].dropna()

p_cut_20 = p_vals.quantile(0.80)  # top 20% (selected)
p_cut_30 = p_vals.quantile(0.70)  # top 30%
p_cut_40 = p_vals.quantile(0.60)  # top 40%

plt.figure(figsize=(7, 5))
plt.hist(p_vals, bins=30, alpha=0.7)

plt.axvline(p_cut_20, linestyle="-", linewidth=2.5, label=f"Top 20% ({p_cut_20:.2f}) [Selected]")
plt.axvline(p_cut_30, linestyle="--", linewidth=1.5, label=f"Top 30% ({p_cut_30:.2f})")
plt.axvline(p_cut_40, linestyle="--", linewidth=1.5, label=f"Top 40% ({p_cut_40:.2f})")

# plt.title("Pitcher: Percentile Thresholds for Success Classification")
plt.xlabel("Composite Score")
plt.ylabel("Frequency")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figure3b_pitcher_percentile_thresholds.png", dpi=300, bbox_inches="tight")
plt.show()