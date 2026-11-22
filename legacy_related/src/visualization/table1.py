import pandas as pd

# 1. 데이터 불러오기
hitter = pd.read_csv("data/final_hitter_fa_dataset_2011_2024_FAonly_clean_posgroup.csv")
pitcher = pd.read_csv("data/final_pitcher_fa_dataset_2011_2024_FAonly_clean.csv")

# 2. 요약 테이블 생성
summary = pd.DataFrame({
    "Group": ["Hitter", "Pitcher"],
    "N": [len(hitter), len(pitcher)],
    
    "Mean WAR": [
        hitter["post_WAR_3y_sum"].mean(),
        pitcher["post_WAR_3y_sum"].mean()
    ],
    
    "Std WAR": [
        hitter["post_WAR_3y_sum"].std(),
        pitcher["post_WAR_3y_sum"].std()
    ],
    
    "Mean Playtime(PA/IP)": [
        hitter["post_PA_3y"].mean(),
        pitcher["post_IP_3y"].mean()
    ],
    
    "Std Playtime(PA/IP)": [
        hitter["post_PA_3y"].std(),
        pitcher["post_IP_3y"].std()
    ]
})

# 3. 소수점 정리
summary = summary.round(2)

print(summary)