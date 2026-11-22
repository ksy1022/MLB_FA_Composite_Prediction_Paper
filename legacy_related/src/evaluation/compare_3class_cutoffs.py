# -*- coding: utf-8 -*-
"""
3-class cutoff comparison
- 실패: post_WAR_3y_sum < 0
- 성공: 양수 구간에서 상위 20% / 30% / 40%
- 보통: 그 사이

비교 항목:
- class distribution
- accuracy
- macro_f1
- weighted_f1

실행 예시:
    python compare_3class_cutoffs.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


# =========================================================
# 0. 경로
# =========================================================
BASE_DIR = Path("data")
HITTER_PATH = BASE_DIR / "final_hitter_fa_dataset_2011_2024_FAonly_clean_posgroup.csv"
PITCHER_PATH = BASE_DIR / "final_pitcher_fa_dataset_2011_2024_FAonly_clean.csv"

RESULT_DIR = BASE_DIR / "results_3class_compare"
RESULT_DIR.mkdir(exist_ok=True)


# =========================================================
# 1. 유틸
# =========================================================
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def print_class_distribution(y: pd.Series, name: str) -> None:
    counts = y.value_counts(dropna=False).sort_index()
    ratios = y.value_counts(normalize=True, dropna=False).sort_index() * 100

    print(f"\n[CLASS DISTRIBUTION] {name}")
    for cls in counts.index:
        print(f"  class={cls}: {counts[cls]} ({ratios[cls]:.2f}%)")


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

    return preprocessor


# =========================================================
# 2. Pitcher role 생성
# =========================================================
def add_pitcher_role(df: pd.DataFrame) -> pd.DataFrame:
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
    return out


# =========================================================
# 3. 3-class 라벨 생성
# =========================================================
def make_3class_label_by_positive_quantile(
    df: pd.DataFrame,
    score_col: str,
    success_top_ratio: float,
    label_name: str = "target_3class"
) -> pd.DataFrame:
    """
    실패: score < 0
    성공: score >= positive score의 (1 - success_top_ratio) 분위수
    보통: 그 사이

    예:
    success_top_ratio=0.30 이면
    양수 구간 내 상위 30%를 성공으로 정의
    """
    out = df.copy()

    score = safe_numeric(out[score_col])

    positive_scores = score[score >= 0].dropna()
    if len(positive_scores) == 0:
        raise ValueError("양수 score가 없습니다. 3-class 기준을 만들 수 없습니다.")

    q = 1.0 - success_top_ratio
    success_cut = positive_scores.quantile(q)

    out[label_name] = 1
    out.loc[score < 0, label_name] = 0
    out.loc[score >= success_cut, label_name] = 2

    return out, float(success_cut)


# =========================================================
# 4. 간단 baseline 모델 평가
# =========================================================
def run_multiclass_experiment(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    experiment_name: str
):
    usable_features = [f for f in features if f in df.columns]
    X = df[usable_features].copy()
    y = df[target_col].copy()

    print("\n" + "=" * 80)
    print(f"[EXPERIMENT] {experiment_name}")
    print("=" * 80)
    print(f"[INFO] rows: {len(df)}")
    print(f"[INFO] features: {len(usable_features)}")
    print_class_distribution(y, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    weighted_f1 = f1_score(y_test, pred, average="weighted")

    print("\n[TEST REPORT]")
    print(classification_report(y_test, pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro F1     : {macro_f1:.4f}")
    print(f"Weighted F1  : {weighted_f1:.4f}")

    return {
        "experiment": experiment_name,
        "n_rows": len(df),
        "n_features": len(usable_features),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


# =========================================================
# 5. main
# =========================================================
def main():
    hitter = pd.read_csv(HITTER_PATH)
    pitcher = pd.read_csv(PITCHER_PATH)
    pitcher = add_pitcher_role(pitcher)

    # 비교용 feature
    hitter_features = [
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

    pitcher_features = [
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

    success_ratios = [0.20, 0.30, 0.40]
    all_results = []

    # -------------------------
    # Hitter
    # -------------------------
    for r in success_ratios:
        temp_df, cut = make_3class_label_by_positive_quantile(
            hitter,
            score_col="post_WAR_3y_sum",
            success_top_ratio=r,
            label_name="h_target_3class"
        )

        print("\n" + "#" * 80)
        print(f"[HITTER] success_top_ratio={r:.2f}, success_cut={cut:.4f}")
        print("#" * 80)

        result = run_multiclass_experiment(
            df=temp_df,
            features=hitter_features,
            target_col="h_target_3class",
            experiment_name=f"hitter_3class_top{int(r*100)}"
        )
        result["success_top_ratio"] = r
        result["success_cut"] = cut
        result["group"] = "hitter"
        all_results.append(result)

    # -------------------------
    # Pitcher
    # -------------------------
    for r in success_ratios:
        temp_df, cut = make_3class_label_by_positive_quantile(
            pitcher,
            score_col="post_WAR_3y_sum",
            success_top_ratio=r,
            label_name="p_target_3class"
        )

        print("\n" + "#" * 80)
        print(f"[PITCHER] success_top_ratio={r:.2f}, success_cut={cut:.4f}")
        print("#" * 80)

        result = run_multiclass_experiment(
            df=temp_df,
            features=pitcher_features,
            target_col="p_target_3class",
            experiment_name=f"pitcher_3class_top{int(r*100)}"
        )
        result["success_top_ratio"] = r
        result["success_cut"] = cut
        result["group"] = "pitcher"
        all_results.append(result)

    # -------------------------
    # 결과 저장
    # -------------------------
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(RESULT_DIR / "compare_3class_cutoffs.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("[FINAL COMPARISON]")
    print("=" * 80)
    print(result_df.sort_values(["group", "accuracy"], ascending=[True, False]))


if __name__ == "__main__":
    main()