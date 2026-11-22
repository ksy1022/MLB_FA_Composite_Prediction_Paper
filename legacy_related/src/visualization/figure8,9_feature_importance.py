import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# 경로 설정
# =========================================================
RESULT_DIR = Path("data/results_threshold_final")

HITTER_IMP_PATH = RESULT_DIR / "hitter_expanded_main_xgb_importance.csv"
PITCHER_SP_IMP_PATH = RESULT_DIR / "pitcher_sp_expanded_main_xgb_importance.csv"
PITCHER_RP_IMP_PATH = RESULT_DIR / "pitcher_rp_expanded_main_xgb_importance.csv"

# =========================================================
# 유틸 함수
# =========================================================
def load_importance_csv(path: Path) -> pd.Series:
    """
    저장된 feature importance csv를 읽어서 Series로 반환
    """
    df = pd.read_csv(path, index_col=0)

    # 1열짜리 dataframe이면 series로 변환
    if isinstance(df, pd.DataFrame):
        if df.shape[1] == 1:
            s = df.iloc[:, 0]
        else:
            # 혹시 여러 컬럼이 있으면 첫 번째 컬럼 사용
            s = df.iloc[:, 0]
    else:
        s = df

    s.index = [str(x).replace("_", " ") for x in s.index]
    return s.sort_values(ascending=False)


def plot_top_importance(
    importance: pd.Series,
    title: str,
    save_path: str,
    top_n: int = 10
):
    """
    단일 importance barh plot
    """
    top_imp = importance.head(top_n).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(top_imp.index, top_imp.values)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pitcher_importance_two_panel(
    sp_importance: pd.Series,
    rp_importance: pd.Series,
    save_path: str,
    top_n: int = 10
):
    """
    SP / RP 2패널 importance plot
    """
    top_sp = sp_importance.head(top_n).sort_values(ascending=True)
    top_rp = rp_importance.head(top_n).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(top_sp.index, top_sp.values)
    axes[0].set_title("Pitcher (SP)")
    axes[0].set_xlabel("Feature Importance")
    axes[0].set_ylabel("Feature")

    axes[1].barh(top_rp.index, top_rp.values)
    axes[1].set_title("Pitcher (RP)")
    axes[1].set_xlabel("Feature Importance")
    axes[1].set_ylabel("Feature")

    # plt.suptitle("Figure 9. Top 10 Feature Importances in Pitcher Models", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# Figure 8: Hitter
# =========================================================
hitter_imp = load_importance_csv(HITTER_IMP_PATH)

plot_top_importance(
    importance=hitter_imp,
    title="Figure 8. Top 10 Feature Importances in Hitter Model",
    save_path="figure8_hitter_feature_importance.png",
    top_n=10
)

# =========================================================
# Figure 9: Pitcher (SP / RP)
# =========================================================
sp_imp = load_importance_csv(PITCHER_SP_IMP_PATH)
rp_imp = load_importance_csv(PITCHER_RP_IMP_PATH)

plot_pitcher_importance_two_panel(
    sp_importance=sp_imp,
    rp_importance=rp_imp,
    save_path="figure9_pitcher_feature_importance.png",
    top_n=10
)