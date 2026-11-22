import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import precision_recall_curve, average_precision_score

warnings.filterwarnings("ignore")

# =========================================================
# 0. 경로
# =========================================================
BASE_DIR = Path("data")
PITCHER_PATH = BASE_DIR / "final_pitcher_fa_dataset_2011_2024_FAonly_clean.csv"

# =========================================================
# 1. 유틸
# =========================================================
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def pick_first_existing(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# =========================================================
# 2. Feature Engineering (Pitcher)
# =========================================================
def engineer_pitcher_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Pos" in out.columns:
        out["Pos"] = out["Pos"].fillna("").astype(str).str.upper()
    else:
        out["Pos"] = ""

    def make_role(pos: str) -> str:
        if "SP" in pos:
            return "SP"
        return "RP"

    out["Role"] = out["Pos"].apply(make_role)
    out["is_SP"] = (out["Role"] == "SP").astype(int)
    out["is_RP"] = (out["Role"] == "RP").astype(int)

    war_last1 = pick_first_existing(out, [
        "pre_WAR_last1y", "pre_WAR_1y", "pre_WAR_last_year", "pre_WAR_y1"
    ])
    ip_last1 = pick_first_existing(out, [
        "pre_IP_last1y", "pre_IP_1y", "pre_IP_last_year", "pre_IP_y1"
    ])
    fip_last1 = pick_first_existing(out, [
        "pre_FIP_last1y", "pre_FIP_1y", "pre_FIP_last_year", "pre_FIP_y1"
    ])
    era_last1 = pick_first_existing(out, [
        "pre_ERA_last1y", "pre_ERA_1y", "pre_ERA_last_year", "pre_ERA_y1"
    ])
    kbb_last1 = pick_first_existing(out, [
        "pre_KBB_last1y", "pre_KBB_1y", "pre_KBB_last_year", "pre_KBB_y1"
    ])

    if war_last1 is None and col_exists(out, "pre_WAR_3y_avg"):
        out["pre_WAR_last1y_proxy"] = safe_numeric(out["pre_WAR_3y_avg"])
        war_last1 = "pre_WAR_last1y_proxy"

    if ip_last1 is None and col_exists(out, "pre_IP_3y"):
        out["pre_IP_last1y_proxy"] = safe_numeric(out["pre_IP_3y"]) / 3.0
        ip_last1 = "pre_IP_last1y_proxy"

    if fip_last1 is None and col_exists(out, "pre_FIP_3y_avg"):
        out["pre_FIP_last1y_proxy"] = safe_numeric(out["pre_FIP_3y_avg"])
        fip_last1 = "pre_FIP_last1y_proxy"

    if era_last1 is None and col_exists(out, "pre_ERA_3y_avg"):
        out["pre_ERA_last1y_proxy"] = safe_numeric(out["pre_ERA_3y_avg"])
        era_last1 = "pre_ERA_last1y_proxy"

    if kbb_last1 is None and col_exists(out, "pre_KBB_3y_avg"):
        out["pre_KBB_last1y_proxy"] = safe_numeric(out["pre_KBB_3y_avg"])
        kbb_last1 = "pre_KBB_last1y_proxy"

    if war_last1 and col_exists(out, "pre_WAR_3y_avg"):
        out["pre_WAR_trend"] = safe_numeric(out[war_last1]) - safe_numeric(out["pre_WAR_3y_avg"])

    if ip_last1 and col_exists(out, "pre_IP_3y"):
        out["pre_IP_last1y_share"] = safe_numeric(out[ip_last1]) / (safe_numeric(out["pre_IP_3y"]) + 1)

    if fip_last1 and col_exists(out, "pre_FIP_3y_avg"):
        out["pre_FIP_trend"] = safe_numeric(out[fip_last1]) - safe_numeric(out["pre_FIP_3y_avg"])

    if era_last1 and col_exists(out, "pre_ERA_3y_avg"):
        out["pre_ERA_trend"] = safe_numeric(out[era_last1]) - safe_numeric(out["pre_ERA_3y_avg"])

    if kbb_last1 and col_exists(out, "pre_KBB_3y_avg"):
        out["pre_KBB_trend"] = safe_numeric(out[kbb_last1]) - safe_numeric(out["pre_KBB_3y_avg"])

    if col_exists(out, "pre_GS_3y_sum") and col_exists(out, "pre_G_3y_sum"):
        out["GS_ratio"] = safe_numeric(out["pre_GS_3y_sum"]) / (safe_numeric(out["pre_G_3y_sum"]) + 1)

    if col_exists(out, "pre_IP_3y") and col_exists(out, "pre_G_3y_sum"):
        out["IP_per_G"] = safe_numeric(out["pre_IP_3y"]) / (safe_numeric(out["pre_G_3y_sum"]) + 1)

    if col_exists(out, "pre_WAR_3y_avg"):
        out["pre_WAR_x_SP"] = safe_numeric(out["pre_WAR_3y_avg"]) * out["is_SP"]

    if col_exists(out, "pre_IP_3y"):
        out["pre_IP_x_SP"] = safe_numeric(out["pre_IP_3y"]) * out["is_SP"]

    if col_exists(out, "pre_KBB_3y_avg"):
        out["pre_KBB_x_RP"] = safe_numeric(out["pre_KBB_3y_avg"]) * out["is_RP"]

    if col_exists(out, "pre_WAR_3y_avg") and col_exists(out, "AAV"):
        out["pre_WAR_x_AAV"] = safe_numeric(out["pre_WAR_3y_avg"]) * safe_numeric(out["AAV"])

    if col_exists(out, "age_at_contract") and col_exists(out, "pre_WAR_3y_avg"):
        out["age_x_pre_WAR"] = safe_numeric(out["age_at_contract"]) * safe_numeric(out["pre_WAR_3y_avg"])

    return out

# =========================================================
# 3. 2-class label 생성
# =========================================================
def create_success_labels(
    df: pd.DataFrame,
    post_war_col: str,
    post_playtime_col: str,
    success_top_ratio: float,
    label_prefix: str = ""
) -> pd.DataFrame:
    out = df.copy()

    score_cols = [post_war_col, post_playtime_col]
    temp = out[score_cols].copy()

    for c in score_cols:
        temp[c] = pd.to_numeric(temp[c], errors="coerce")
        temp[c] = temp[c].fillna(temp[c].median())

    # z-score
    z = (temp - temp.mean()) / temp.std()

    out[f"{label_prefix}z_post_WAR"] = z.iloc[:, 0]
    out[f"{label_prefix}z_post_playtime"] = z.iloc[:, 1]

    out[f"{label_prefix}composite_score"] = (
        out[f"{label_prefix}z_post_WAR"] + out[f"{label_prefix}z_post_playtime"]
    )

    success_cut = out[f"{label_prefix}composite_score"].quantile(1.0 - success_top_ratio)
    out[f"{label_prefix}success_main"] = (
        out[f"{label_prefix}composite_score"] >= success_cut
    ).astype(int)

    return out

# =========================================================
# 4. 전처리
# =========================================================
def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor

# =========================================================
# 5. RP expanded features
# =========================================================
PITCHER_RP_EXPANDED_FEATURES = [
    "age_at_contract",
    "pre_IP_3y",
    "pre_WAR_3y_avg",
    "pre_ERA_3y_avg",
    "pre_FIP_3y_avg",
    "pre_KBB_3y_avg",
    "pre_G_3y_sum",
    "AAV",
    "Yrs",
    "pre_WAR_trend",
    "pre_IP_last1y_share",
    "pre_FIP_trend",
    "pre_ERA_trend",
    "pre_KBB_trend",
    "IP_per_G",
    "pre_KBB_x_RP",
    "pre_WAR_x_AAV",
    "age_x_pre_WAR",
]

# =========================================================
# 6. 데이터 로드 및 RP 추출
# =========================================================
df = pd.read_csv(PITCHER_PATH)
df = engineer_pitcher_features(df)
df = create_success_labels(
    df=df,
    post_war_col="post_WAR_3y_sum",
    post_playtime_col="post_IP_3y",
    success_top_ratio=0.20,
    label_prefix="p_"
)

rp_df = df[df["Role"] == "RP"].copy()

usable_features = [f for f in PITCHER_RP_EXPANDED_FEATURES if f in rp_df.columns]
X = rp_df[usable_features].copy()
y = rp_df["p_success_main"].copy()

# train / valid / test 분할 (기존 실험과 동일)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp,
    test_size=0.25,   # 최종 60 / 20 / 20
    stratify=y_temp,
    random_state=42
)

preprocessor = build_preprocessor(X_train)

# =========================================================
# 7. 모델 정의
# =========================================================
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
scale_pos_weight = neg / pos if pos > 0 else 1.0

models = {
    "Logistic Regression": LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=3000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
}

# =========================================================
# 8. PR curve 그리기
# =========================================================
plt.figure(figsize=(7, 5))

for model_name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_score = pipe.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    # 핵심 변경
    plt.step(recall, precision, where='post',
             linewidth=2, alpha=0.9,
             label=f"{model_name} (PR-AUC = {pr_auc:.3f})")

    # 포인트 표시
    plt.scatter(recall, precision, s=15)

# baseline
baseline = y_test.mean()
plt.axhline(y=baseline, linestyle="--", linewidth=1.5,
            label=f"Baseline = {baseline:.3f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
# plt.title("Precision-Recall Curves for Pitcher (RP) Models")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figure7_rp_pr_curve_clean.png", dpi=300)
plt.show()