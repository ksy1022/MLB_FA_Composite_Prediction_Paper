# -*- coding: utf-8 -*-
"""
FA Success Prediction Experiment (Final Tuning Version)
- Hitter / Pitcher 분리
- AAV 제거된 Composite Success Score 사용
- Train / Valid / Test 분할
- Validation 기반 threshold 최적화
- GridSearchCV 기반 간단한 하이퍼파라미터 튜닝
- Logistic / RandomForest / XGBoost 비교
- Accuracy, ROC-AUC, PR-AUC, Precision, Recall, F1 출력

실행 예시:
    python fa_model_experiment.py

필수 패키지:
    pip install pandas numpy scikit-learn xgboost matplotlib
"""

import warnings
from pathlib import Path

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
from sklearn.ensemble import RandomForestClassifier
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
BASE_DIR = Path("data")
HITTER_PATH = BASE_DIR / "final_hitter_fa_dataset_2011_2024_FAonly_clean_posgroup.csv"
PITCHER_PATH = BASE_DIR / "final_pitcher_fa_dataset_2011_2024_FAonly_clean.csv"

RESULT_DIR = BASE_DIR / "results_tuned"
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


def find_best_threshold_by_accuracy(y_true: pd.Series, y_proba: np.ndarray):
    best_t = 0.50
    best_acc = -1.0

    for t in np.linspace(0.30, 0.70, 81):
        pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_t = t

    return float(best_t), float(best_acc)


# =========================================================
# 2. Composite Score 및 Success Label 생성
# =========================================================
def create_success_labels(
    df: pd.DataFrame,
    post_war_col: str,
    post_playtime_col: str,
    pre_war_col: str,
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

    score_median = out[f"{label_prefix}composite_score"].median()
    out[f"{label_prefix}success_main"] = (
        out[f"{label_prefix}composite_score"] >= score_median
    ).astype(int)

    out[f"{label_prefix}success_retention"] = (
        out[post_war_col] >= out[pre_war_col]
    ).astype(int)

    score_p60 = out[f"{label_prefix}composite_score"].quantile(0.60)
    out[f"{label_prefix}success_p60"] = (
        out[f"{label_prefix}composite_score"] >= score_p60
    ).astype(int)

    print(f"\n[{label_prefix}] Composite Score Summary")
    print(out[f"{label_prefix}composite_score"].describe())
    print(f"Median(score): {score_median:.4f}")
    print(f"60th percentile(score): {score_p60:.4f}")

    return out


# =========================================================
# 3. 데이터 로드 및 전처리
# =========================================================
def load_and_prepare_hitter(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    pre_war_col = "pre_WAR_3y_sum" if "pre_WAR_3y_sum" in df.columns else "pre_WAR_3y_avg"

    df = create_success_labels(
        df=df,
        post_war_col="post_WAR_3y_sum",
        post_playtime_col="post_PA_3y",
        pre_war_col=pre_war_col,
        label_prefix="h_"
    )
    return df


def load_and_prepare_pitcher(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Pos" in df.columns:
        df["Pos"] = df["Pos"].fillna("").astype(str).str.upper()
    else:
        df["Pos"] = ""

    def make_role(pos: str) -> str:
        if "SP" in pos:
            return "SP"
        return "RP"

    df["Role"] = df["Pos"].apply(make_role)

    pre_war_col = "pre_WAR_3y_sum" if "pre_WAR_3y_sum" in df.columns else "pre_WAR_3y_avg"

    df = create_success_labels(
        df=df,
        post_war_col="post_WAR_3y_sum",
        post_playtime_col="post_IP_3y",
        pre_war_col=pre_war_col,
        label_prefix="p_"
    )

    print("\n=== PITCHER ROLE CHECK ===")
    print(df["Role"].value_counts(dropna=False))

    return df


# =========================================================
# 4. Feature Set 정의
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
]


# =========================================================
# 5. 전처리 파이프라인
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
# 6. 모델 정의 + 튜닝 공간
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

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model, param_grid


# =========================================================
# 7. 학습 및 평가 (Train / Valid / Test + Threshold Tuning)
# =========================================================
def run_experiment(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    experiment_name: str,
    model_names=("logistic", "rf", "xgb")
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

    # 1차: test 분리
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 2차: temp -> train / valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp,
        test_size=0.25,   # 전체 기준 0.2
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

        # validation set으로 threshold 최적화
        valid_proba = best_pipe.predict_proba(X_valid)[:, 1]
        best_threshold, best_valid_acc = find_best_threshold_by_accuracy(y_valid, valid_proba)
        print(f"[BEST THRESHOLD on VALID] {best_threshold:.3f} (valid acc={best_valid_acc:.4f})")

        # test 평가
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
# 8. 메인 실행
# =========================================================
def main():
    if not HITTER_PATH.exists():
        raise FileNotFoundError(f"Hitter file not found: {HITTER_PATH}")

    if not PITCHER_PATH.exists():
        raise FileNotFoundError(f"Pitcher file not found: {PITCHER_PATH}")

    hitter = load_and_prepare_hitter(HITTER_PATH)
    pitcher = load_and_prepare_pitcher(PITCHER_PATH)

    hitter[[
        "h_composite_score", "h_success_main",
        "h_success_retention", "h_success_p60"
    ]].to_csv(RESULT_DIR / "hitter_labels_preview.csv", index=False, encoding="utf-8-sig")

    pitcher[[
        "p_composite_score", "p_success_main",
        "p_success_retention", "p_success_p60"
    ]].to_csv(RESULT_DIR / "pitcher_labels_preview.csv", index=False, encoding="utf-8-sig")

    print_class_distribution(hitter["h_success_main"], "Hitter Main Success")
    print_class_distribution(hitter["h_success_retention"], "Hitter Retention Success")
    print_class_distribution(hitter["h_success_p60"], "Hitter P60 Success")

    print_class_distribution(pitcher["p_success_main"], "Pitcher Main Success")
    print_class_distribution(pitcher["p_success_retention"], "Pitcher Retention Success")
    print_class_distribution(pitcher["p_success_p60"], "Pitcher P60 Success")

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
    # Pitcher Main
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
    # Pitcher Robustness
    # -----------------------------
    results.append(run_experiment(
        df=pitcher,
        features=PITCHER_EXPANDED_FEATURES,
        target_col="p_success_retention",
        experiment_name="pitcher_expanded_retention_robustness"
    ))

    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(RESULT_DIR / "all_experiment_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("[FINAL SUMMARY]")
    print("=" * 80)
    print(all_results.sort_values(["experiment", "accuracy"], ascending=[True, False]))


if __name__ == "__main__":
    main()