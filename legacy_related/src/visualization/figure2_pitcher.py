import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
hitter = pd.read_csv("data/final_hitter_fa_dataset_2011_2024_FAonly_clean_posgroup.csv")
pitcher = pd.read_csv("data/final_pitcher_fa_dataset_2011_2024_FAonly_clean.csv")

# z-score 함수
def zscore(s):
    s = s.fillna(s.median())
    return (s - s.mean()) / s.std()

# composite score 생성
pitcher["z_WAR"] = zscore(pitcher["post_WAR_3y_sum"])
pitcher["z_playtime"] = zscore(pitcher["post_IP_3y"])
pitcher["composite_score"] = pitcher["z_WAR"] + pitcher["z_playtime"]

# Figure
plt.figure(figsize=(8,5))

plt.hist(pitcher["composite_score"], bins=30, alpha=0.6, label="Pitcher")

plt.xlabel("Composite Score")
plt.ylabel("Frequency")
# plt.title("Distribution of Composite Performance Score")
plt.legend()

plt.tight_layout()
# plt.savefig("figure2_composite_distribution.png", dpi=300)
plt.show()

