# -*- coding: utf-8 -*-
"""
FA Success Prediction Experiment
- Hitter / Pitcher 분리
- Composite Success Score (AAV 제외)
- Train / Valid / Test 분리
- Validation 기반 threshold 최적화
- Logistic / RandomForest / XGBoost / Stacking 비교
- Pitcher 전체 + SP / RP 분리 실험
- Accuracy, ROC-AUC, PR-AUC, Precision, Recall, F1 출력

실행 예시:
    python fa_model_experiment.py

필수 패키지:
    pip install pandas numpy scikit-learn xgboost matplotlib
"""

import warnings
from pathlib import Path
import joblib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

warnings.filterwarnings("ignore")


# =========================================================
# 0. 경로 설정
# =========================================================
BASE_DIR = Path("data/processed")
HITTER_PATH = BASE_DIR / "final_hitter_fa_dataset_2011_2024_FAonly_clean_posgroup.csv"
PITCHER_PATH = BASE_DIR / "final_pitcher_fa_dataset_2011_2024_FAonly_clean.csv"

RESULT_DIR = Path("outputs/final_2class")
RESULT_DIR.mkdir(exist_ok=True)


# =========================================================
# 1. 유틸 함수
# =========================================================
def print_class_distribution(y: pd.Series, name: str) -> None:
    counts = y.value_counts(dropna=False).sort_index()
    ratios = y.value_counts(normalize=True, dropna=False).sort_index() * 100

    print(f"\n[CLASS DISTRIBUTION] {name}")
    for cls in counts.index:
        print(f"  class={cls}: {counts[cls]} ({ratios[cls]:.2f}%)")


def save_feature_importance(model, feature_names, save_path: Path, title: str):
    if not hasattr(model, "feature_importances_"):
        return

    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    importance.to_csv(save_path.with_suffix(".csv"), encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    importance.head(20).sort_values().plot(kind="barh")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=200)
    plt.close("all")


def extract_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features):
    feature_names = []

    if numeric_features:
        feature_names.extend(numeric_features)

    if categorical_features:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(cat_names)

    return feature_names


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def pick_first_existing(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_best_threshold_by_accuracy(y_true, y_proba):
    best_threshold = 0.50
    best_acc = -1.0

    for t in np.arange(0.10, 0.91, 0.01):
        pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y_true, pred)

        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    return float(best_threshold), float(best_acc)


# =========================================================
# 2. Composite Score 및 Success Label 생성
#    AAV는 결과변수에서 제외
# =========================================================
def create_success_labels(
    df: pd.DataFrame,
    post_war_col: str,
    post_playtime_col: str,
    pre_war_col: str,
    success_top_ratio: float,
    label_prefix: str = ""
) -> pd.DataFrame:
    out = df.copy()

    score_cols = [post_war_col, post_playtime_col]

    temp = out[score_cols].copy()
    for c in score_cols:
        temp[c] = temp[c].fillna(temp[c].median())

    scaler = StandardScaler()
    z = scaler.fit_transform(temp)

    out[f"{label_prefix}z_post_WAR"] = z[:, 0]
    out[f"{label_prefix}z_post_playtime"] = z[:, 1]

    out[f"{label_prefix}composite_score"] = (
        out[f"{label_prefix}z_post_WAR"]
        + out[f"{label_prefix}z_post_playtime"]
    )

    # main = top percentile
    success_cut = out[f"{label_prefix}composite_score"].quantile(1.0 - success_top_ratio)
    out[f"{label_prefix}success_main"] = (
        out[f"{label_prefix}composite_score"] >= success_cut
    ).astype(int)

    # robustness 1: retention
    out[f"{label_prefix}success_retention"] = (
        out[post_war_col] >= out[pre_war_col]
    ).astype(int)

    # robustness 2: median
    score_median = out[f"{label_prefix}composite_score"].median()
    out[f"{label_prefix}success_median"] = (
        out[f"{label_prefix}composite_score"] >= score_median
    ).astype(int)

    print(f"\n[{label_prefix}] Composite Score Summary")
    print(out[f"{label_prefix}composite_score"].describe())
    print(f"Main success cut (top {int(success_top_ratio*100)}%): {success_cut:.4f}")
    print(f"Median(score): {score_median:.4f}")

    return out


# =========================================================
# 3. Feature Engineering
# =========================================================
def engineer_hitter_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    war_last1 = pick_first_existing(out, [
        "pre_WAR_last1y", "pre_WAR_1y", "pre_WAR_last_year", "pre_WAR_y1"
    ])
    ops_last1 = pick_first_existing(out, [
        "pre_OPSplus_last1y", "pre_OPSplus_1y", "pre_OPSplus_last_year", "pre_OPSplus_y1"
    ])
    pa_last1 = pick_first_existing(out, [
        "pre_PA_last1y", "pre_PA_1y", "pre_PA_last_year", "pre_PA_y1"
    ])
    bb_last1 = pick_first_existing(out, [
        "pre_BB_rate_last1y", "pre_BB_rate_1y", "pre_BB_rate_last_year", "pre_BB_rate_y1"
    ])
    k_last1 = pick_first_existing(out, [
        "pre_K_rate_last1y", "pre_K_rate_1y", "pre_K_rate_last_year", "pre_K_rate_y1"
    ])

    if war_last1 is None and col_exists(out, "pre_WAR_3y_avg"):
        out["pre_WAR_last1y_proxy"] = safe_numeric(out["pre_WAR_3y_avg"])
        war_last1 = "pre_WAR_last1y_proxy"

    if ops_last1 is None and col_exists(out, "pre_OPSplus_3y_avg"):
        out["pre_OPSplus_last1y_proxy"] = safe_numeric(out["pre_OPSplus_3y_avg"])
        ops_last1 = "pre_OPSplus_last1y_proxy"

    if pa_last1 is None and col_exists(out, "pre_PA_3y"):
        out["pre_PA_last1y_proxy"] = safe_numeric(out["pre_PA_3y"]) / 3.0
        pa_last1 = "pre_PA_last1y_proxy"

    if bb_last1 is None and col_exists(out, "pre_BB_rate_3y_avg"):
        out["pre_BB_rate_last1y_proxy"] = safe_numeric(out["pre_BB_rate_3y_avg"])
        bb_last1 = "pre_BB_rate_last1y_proxy"

    if k_last1 is None and col_exists(out, "pre_K_rate_3y_avg"):
        out["pre_K_rate_last1y_proxy"] = safe_numeric(out["pre_K_rate_3y_avg"])
        k_last1 = "pre_K_rate_last1y_proxy"

    if war_last1 and col_exists(out, "pre_WAR_3y_avg"):
        out["pre_WAR_trend"] = safe_numeric(out[war_last1]) - safe_numeric(out["pre_WAR_3y_avg"])

    if ops_last1 and col_exists(out, "pre_OPSplus_3y_avg"):
        out["pre_OPSplus_trend"] = safe_numeric(out[ops_last1]) - safe_numeric(out["pre_OPSplus_3y_avg"])

    if bb_last1 and col_exists(out, "pre_BB_rate_3y_avg"):
        out["pre_BB_rate_trend"] = safe_numeric(out[bb_last1]) - safe_numeric(out["pre_BB_rate_3y_avg"])

    if k_last1 and col_exists(out, "pre_K_rate_3y_avg"):
        out["pre_K_rate_trend"] = safe_numeric(out[k_last1]) - safe_numeric(out["pre_K_rate_3y_avg"])

    if pa_last1 and col_exists(out, "pre_PA_3y"):
        out["pre_PA_last1y_share"] = safe_numeric(out[pa_last1]) / (safe_numeric(out["pre_PA_3y"]) + 1)

    if col_exists(out, "pre_WAR_3y_std"):
        out["pre_WAR_consistency"] = 1.0 / (safe_numeric(out["pre_WAR_3y_std"]) + 1.0)

    if col_exists(out, "pre_WAR_3y_avg") and col_exists(out, "AAV"):
        out["pre_WAR_x_AAV"] = safe_numeric(out["pre_WAR_3y_avg"]) * safe_numeric(out["AAV"])

    if col_exists(out, "age_at_contract") and col_exists(out, "pre_WAR_3y_avg"):
        out["age_x_pre_WAR"] = safe_numeric(out["age_at_contract"]) * safe_numeric(out["pre_WAR_3y_avg"])

    if col_exists(out, "Pos_group"):
        out["is_C"] = (out["Pos_group"] == "C").astype(int)
        out["is_IF"] = (out["Pos_group"] == "IF").astype(int)
        out["is_OF"] = (out["Pos_group"] == "OF").astype(int)
        out["is_DH"] = (out["Pos_group"] == "DH").astype(int)

        if col_exists(out, "pre_WAR_3y_avg"):
            out["pre_WAR_x_is_C"] = safe_numeric(out["pre_WAR_3y_avg"]) * out["is_C"]
            out["pre_WAR_x_is_DH"] = safe_numeric(out["pre_WAR_3y_avg"]) * out["is_DH"]

    print("\n=== ENGINEERED HITTER COLS CHECK ===")
    check_cols = [
        "pre_WAR_trend", "pre_OPSplus_trend", "pre_BB_rate_trend",
        "pre_K_rate_trend", "pre_PA_last1y_share", "pre_WAR_consistency",
        "pre_WAR_x_AAV", "age_x_pre_WAR",
        "is_C", "is_IF", "is_OF", "is_DH",
        "pre_WAR_x_is_C", "pre_WAR_x_is_DH"
    ]
    print([c for c in check_cols if c in out.columns])

    return out


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

    print("\n=== ENGINEERED PITCHER COLS CHECK ===")
    check_cols = [
        "is_SP", "is_RP", "pre_WAR_trend", "pre_IP_last1y_share",
        "pre_FIP_trend", "pre_ERA_trend", "pre_KBB_trend",
        "GS_ratio", "IP_per_G", "pre_WAR_x_SP", "pre_IP_x_SP",
        "pre_KBB_x_RP", "pre_WAR_x_AAV", "age_x_pre_WAR"
    ]
    print([c for c in check_cols if c in out.columns])

    return out


# =========================================================
# 4. 데이터 로드 및 준비
# =========================================================
def load_and_prepare_hitter(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = engineer_hitter_features(df)

    pre_war_col = "pre_WAR_3y_sum" if "pre_WAR_3y_sum" in df.columns else "pre_WAR_3y_avg"

    df = create_success_labels(
        df=df,
        post_war_col="post_WAR_3y_sum",
        post_playtime_col="post_PA_3y",
        pre_war_col=pre_war_col,
        success_top_ratio=0.40,
        label_prefix="h_"
    )
    return df


def load_and_prepare_pitcher(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = engineer_pitcher_features(df)

    pre_war_col = "pre_WAR_3y_sum" if "pre_WAR_3y_sum" in df.columns else "pre_WAR_3y_avg"

    df = create_success_labels(
        df=df,
        post_war_col="post_WAR_3y_sum",
        post_playtime_col="post_IP_3y",
        pre_war_col=pre_war_col,
        success_top_ratio=0.20,
        label_prefix="p_"
)

    print("\n=== PITCHER ROLE CHECK ===")
    print(df["Role"].value_counts(dropna=False))

    return df


# =========================================================
# 5. Feature Set 정의
# =========================================================
HITTER_BASELINE_FEATURES = [
    "Pos_group",
    "age_at_contract",
    "pre_WAR_3y_avg",
    "AAV",
    "Yrs",
]

HITTER_EXPANDED_FEATURES = [
    "Pos_group",
    "age_at_contract",
    "pre_PA_3y",
    "pre_WAR_3y_avg",
    "pre_WAR_3y_std",
    "pre_OPSplus_3y_avg",
    "pre_BB_rate_3y_avg",
    "pre_K_rate_3y_avg",
    "AAV",
    "Yrs",
    "pre_WAR_trend",
    "pre_OPSplus_trend",
    "pre_BB_rate_trend",
    "pre_K_rate_trend",
    "pre_PA_last1y_share",
    "pre_WAR_consistency",
    "pre_WAR_x_AAV",
    "age_x_pre_WAR",
    "is_C",
    "is_IF",
    "is_OF",
    "is_DH",
    "pre_WAR_x_is_C",
    "pre_WAR_x_is_DH",
]

PITCHER_BASELINE_FEATURES = [
    "Role",
    "age_at_contract",
    "pre_WAR_3y_avg",
    "AAV",
    "Yrs",
]

PITCHER_EXPANDED_FEATURES = [
    "Role",
    "age_at_contract",
    "pre_IP_3y",
    "pre_WAR_3y_avg",
    "pre_ERA_3y_avg",
    "pre_FIP_3y_avg",
    "pre_KBB_3y_avg",
    "pre_GS_3y_sum",
    "pre_G_3y_sum",
    "AAV",
    "Yrs",
    "is_SP",
    "is_RP",
    "pre_WAR_trend",
    "pre_IP_last1y_share",
    "pre_FIP_trend",
    "pre_ERA_trend",
    "pre_KBB_trend",
    "GS_ratio",
    "IP_per_G",
    "pre_WAR_x_SP",
    "pre_IP_x_SP",
    "pre_KBB_x_RP",
    "pre_WAR_x_AAV",
    "age_x_pre_WAR",
]

PITCHER_SP_BASELINE_FEATURES = [
    "age_at_contract",
    "pre_WAR_3y_avg",
    "AAV",
    "Yrs",
]

PITCHER_SP_EXPANDED_FEATURES = [
    "age_at_contract",
    "pre_IP_3y",
    "pre_WAR_3y_avg",
    "pre_ERA_3y_avg",
    "pre_FIP_3y_avg",
    "pre_KBB_3y_avg",
    "pre_GS_3y_sum",
    "pre_G_3y_sum",
    "AAV",
    "Yrs",
    "pre_WAR_trend",
    "pre_IP_last1y_share",
    "pre_FIP_trend",
    "pre_ERA_trend",
    "pre_KBB_trend",
    "GS_ratio",
    "IP_per_G",
    "pre_WAR_x_SP",
    "pre_IP_x_SP",
    "pre_WAR_x_AAV",
    "age_x_pre_WAR",
]

PITCHER_RP_BASELINE_FEATURES = [
    "age_at_contract",
    "pre_WAR_3y_avg",
    "AAV",
    "Yrs",
]

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
# 6. 전처리
# =========================================================
def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
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

    return preprocessor, numeric_features, categorical_features


# =========================================================
# 7. 모델 + Grid 정의
# =========================================================
def get_model_and_param_grid(model_name: str, y_train: pd.Series):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    if model_name == "logistic":
        model = LogisticRegression(
            max_iter=3000,
            random_state=42
        )
        param_grid = {
            "model__C": [0.01, 0.1, 1.0, 3.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"],
            "model__class_weight": [None, "balanced"]
        }

    elif model_name == "rf":
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
        param_grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [4, 8, None],
            "model__min_samples_leaf": [1, 2, 4],
            "model__min_samples_split": [2, 5],
            "model__max_features": ["sqrt", "log2"]
        }

    elif model_name == "xgb":
        model = XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight
        )
        param_grid = {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.03, 0.05, 0.10],
            "model__max_depth": [3, 4, 5],
            "model__subsample": [0.8, 0.9],
            "model__colsample_bytree": [0.8, 0.9]
        }

    elif model_name == "stack":
        base_lr = LogisticRegression(
            C=0.1,
            solver="lbfgs",
            max_iter=3000,
            random_state=42
        )

        base_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=2,
            min_samples_split=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )

        base_xgb = XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight
        )

        final_est = LogisticRegression(
            max_iter=3000,
            random_state=42
        )

        model = StackingClassifier(
            estimators=[
                ("lr", base_lr),
                ("rf", base_rf),
                ("xgb", base_xgb),
            ],
            final_estimator=final_est,
            stack_method="predict_proba",
            cv=3,
            n_jobs=-1,
            passthrough=False
        )

        param_grid = {
            "model__final_estimator__C": [0.01, 0.1, 1.0, 3.0, 10.0],
            "model__final_estimator__solver": ["lbfgs", "liblinear"]
        }

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model, param_grid


# =========================================================
# 8. 실험 함수
# =========================================================
def run_experiment(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    experiment_name: str,
    model_names=("logistic", "rf", "xgb", "stack")
):
    print("\n" + "=" * 80)
    print(f"[EXPERIMENT] {experiment_name}")
    print("=" * 80)

    usable_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]

    if missing_features:
        print(f"[WARN] Missing features excluded: {missing_features}")

    X = df[usable_features].copy()
    y = df[target_col].copy()

    print(f"[INFO] Number of rows: {len(df):,}")
    print(f"[INFO] Number of features: {len(usable_features)}")
    print(f"[INFO] Target: {target_col}")
    print_class_distribution(y, target_col)

    # train / valid / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp,
        test_size=0.25,   # 60 / 20 / 20
        stratify=y_temp,
        random_state=42
    )

    print(f"[INFO] Train: {len(X_train):,}, Valid: {len(X_valid):,}, Test: {len(X_test):,}")

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    result_rows = []

    for model_name in model_names:
        print("\n" + "-" * 60)
        print(f"[MODEL] {model_name}")

        base_model, param_grid = get_model_and_param_grid(model_name, y_train)

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", base_model)
            ]
        )

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)

        best_pipe = grid.best_estimator_
        print(f"[BEST PARAMS] {grid.best_params_}")

        # valid에서 threshold 결정
        valid_proba = best_pipe.predict_proba(X_valid)[:, 1]
        best_threshold, best_valid_acc = find_best_threshold_by_accuracy(y_valid, valid_proba)
        print(f"[BEST THRESHOLD on VALID] {best_threshold:.3f} (valid acc={best_valid_acc:.4f})")

        # test는 평가만
        test_proba = best_pipe.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= best_threshold).astype(int)

        accuracy = accuracy_score(y_test, test_pred)
        pr_auc = average_precision_score(y_test, test_proba)
        roc_auc = roc_auc_score(y_test, test_proba)
        precision = precision_score(y_test, test_pred, zero_division=0)
        recall = recall_score(y_test, test_pred, zero_division=0)
        f1 = f1_score(y_test, test_pred, zero_division=0)

        print(classification_report(y_test, test_pred, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, test_pred))
        print(f"Accuracy          : {accuracy:.4f}")
        print(f"ROC-AUC           : {roc_auc:.4f}")
        print(f"PR-AUC            : {pr_auc:.4f}")
        print(f"Precision(success): {precision:.4f}")
        print(f"Recall(success)   : {recall:.4f}")
        print(f"F1(success)       : {f1:.4f}")

        result_rows.append({
            "experiment": experiment_name,
            "model": model_name,
            "target": target_col,
            "n_features": len(usable_features),
            "best_threshold": best_threshold,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision_success": precision,
            "recall_success": recall,
            "f1_success": f1
        })

        # SP 최고 모델 저장
        if experiment_name == "pitcher_sp_expanded_main" and model_name == "stack":
            joblib.dump(best_pipe, RESULT_DIR / "pitcher_sp_expanded_stack.pkl")
            print(f"[SAVED MODEL] {RESULT_DIR / 'pitcher_sp_expanded_stack.pkl'}")

        fitted_preprocessor = best_pipe.named_steps["preprocessor"]
        fitted_model = best_pipe.named_steps["model"]

        try:
            feature_names = extract_feature_names_from_preprocessor(
                fitted_preprocessor,
                numeric_features=numeric_features,
                categorical_features=categorical_features
            )

            if model_name in ["rf", "xgb"]:
                save_feature_importance(
                    model=fitted_model,
                    feature_names=feature_names,
                    save_path=RESULT_DIR / f"{experiment_name}_{model_name}_importance",
                    title=f"{experiment_name} - {model_name} Feature Importance"
                )
        except Exception as e:
            print(f"[WARN] Feature importance save failed: {e}")

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(RESULT_DIR / f"{experiment_name}_metrics.csv", index=False, encoding="utf-8-sig")
    return result_df


# =========================================================
# 9. main
# =========================================================
def main():
    if not HITTER_PATH.exists():
        raise FileNotFoundError(f"Hitter file not found: {HITTER_PATH}")

    if not PITCHER_PATH.exists():
        raise FileNotFoundError(f"Pitcher file not found: {PITCHER_PATH}")

    hitter = load_and_prepare_hitter(HITTER_PATH)
    pitcher = load_and_prepare_pitcher(PITCHER_PATH)

    pitcher_sp = pitcher[pitcher["Role"] == "SP"].copy()
    pitcher_rp = pitcher[pitcher["Role"] == "RP"].copy()

    print("\n=== PITCHER SUBGROUP CHECK ===")
    print(f"SP rows: {len(pitcher_sp)}")
    print(f"RP rows: {len(pitcher_rp)}")

    hitter[[
        "h_composite_score", "h_success_main",
        "h_success_retention", "h_success_median"
    ]].to_csv(RESULT_DIR / "hitter_labels_preview.csv", index=False, encoding="utf-8-sig")

    pitcher[[
        "p_composite_score", "p_success_main",
        "p_success_retention", "p_success_median"
    ]].to_csv(RESULT_DIR / "pitcher_labels_preview.csv", index=False, encoding="utf-8-sig")

    print_class_distribution(hitter["h_success_main"], "Hitter Main Success")
    print_class_distribution(hitter["h_success_retention"], "Hitter Retention Success")
    print_class_distribution(hitter["h_success_median"], "Hitter Median Robustness")

    print_class_distribution(pitcher["p_success_main"], "Pitcher Main Success")
    print_class_distribution(pitcher["p_success_retention"], "Pitcher Retention Success")
    print_class_distribution(pitcher["p_success_median"], "Pitcher Median Robustness")

    results = []

    # -----------------------------
    # Hitter Main
    # -----------------------------
    results.append(run_experiment(
        df=hitter,
        features=HITTER_BASELINE_FEATURES,
        target_col="h_success_main",
        experiment_name="hitter_baseline_main"
    ))

    results.append(run_experiment(
        df=hitter,
        features=HITTER_EXPANDED_FEATURES,
        target_col="h_success_main",
        experiment_name="hitter_expanded_main"
    ))

    # -----------------------------
    # Hitter Robustness
    # -----------------------------
    results.append(run_experiment(
        df=hitter,
        features=HITTER_EXPANDED_FEATURES,
        target_col="h_success_retention",
        experiment_name="hitter_expanded_retention_robustness"
    ))

    # -----------------------------
    # Pitcher Main (전체)
    # -----------------------------
    results.append(run_experiment(
        df=pitcher,
        features=PITCHER_BASELINE_FEATURES,
        target_col="p_success_main",
        experiment_name="pitcher_baseline_main"
    ))

    results.append(run_experiment(
        df=pitcher,
        features=PITCHER_EXPANDED_FEATURES,
        target_col="p_success_main",
        experiment_name="pitcher_expanded_main"
    ))

    # -----------------------------
    # Pitcher Robustness (전체)
    # -----------------------------
    results.append(run_experiment(
        df=pitcher,
        features=PITCHER_EXPANDED_FEATURES,
        target_col="p_success_retention",
        experiment_name="pitcher_expanded_retention_robustness"
    ))

    # -----------------------------
    # Pitcher SP Main
    # -----------------------------
    if len(pitcher_sp) >= 50:
        results.append(run_experiment(
            df=pitcher_sp,
            features=PITCHER_SP_BASELINE_FEATURES,
            target_col="p_success_main",
            experiment_name="pitcher_sp_baseline_main"
        ))

        results.append(run_experiment(
            df=pitcher_sp,
            features=PITCHER_SP_EXPANDED_FEATURES,
            target_col="p_success_main",
            experiment_name="pitcher_sp_expanded_main"
        ))

    # -----------------------------
    # Pitcher RP Main
    # -----------------------------
    if len(pitcher_rp) >= 50:
        results.append(run_experiment(
            df=pitcher_rp,
            features=PITCHER_RP_BASELINE_FEATURES,
            target_col="p_success_main",
            experiment_name="pitcher_rp_baseline_main"
        ))

        results.append(run_experiment(
            df=pitcher_rp,
            features=PITCHER_RP_EXPANDED_FEATURES,
            target_col="p_success_main",
            experiment_name="pitcher_rp_expanded_main"
        ))

    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(RESULT_DIR / "all_experiment_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("[FINAL SUMMARY]")
    print("=" * 80)
    print(all_results.sort_values(["experiment", "accuracy"], ascending=[True, False]))


if __name__ == "__main__":
    main()