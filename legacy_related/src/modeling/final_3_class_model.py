# -*- coding: utf-8 -*-
"""
Final 3-class model
- Hitter: success = top 40% among non-negative post_WAR_3y_sum
- Pitcher: success = top 20% among non-negative post_WAR_3y_sum
- failure = post_WAR_3y_sum < 0
- middle = otherwise

모델:
- Logistic Regression
- Random Forest
- XGBoost

평가지표:
- accuracy
- macro_f1
- weighted_f1
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# 0. 경로
# =========================================================
BASE_DIR = Path("data/processed")
HITTER_PATH = BASE_DIR / "final_hitter_fa_dataset_2011_2024_FAonly_clean_posgroup.csv"
PITCHER_PATH = BASE_DIR / "final_pitcher_fa_dataset_2011_2024_FAonly_clean.csv"

RESULT_DIR = Path("outputs/final_3class")
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
# 3. 최종 3-class 라벨 생성
# =========================================================
def make_zscore(df, war_col, volume_col, new_col):
    df = df.copy()

    if war_col not in df.columns:
        raise KeyError(f"[make_zscore] war_col 없음: {war_col}")
    if volume_col not in df.columns:
        raise KeyError(f"[make_zscore] volume_col 없음: {volume_col}")

    war = pd.to_numeric(df[war_col], errors="coerce")
    vol = pd.to_numeric(df[volume_col], errors="coerce")

    war_std = war.std()
    vol_std = vol.std()

    if pd.isna(war_std) or war_std == 0:
        raise ValueError(f"{war_col}의 표준편차가 0이거나 계산할 수 없습니다.")
    if pd.isna(vol_std) or vol_std == 0:
        raise ValueError(f"{volume_col}의 표준편차가 0이거나 계산할 수 없습니다.")

    war_z = (war - war.mean()) / war_std
    vol_z = (vol - vol.mean()) / vol_std

    df[new_col] = war_z + vol_z
    return df

def make_3class_label(
    df: pd.DataFrame,
    score_col: str,
    success_top_ratio: float,
    label_name: str
):
    """
    실패: score < 0 → class 0
    보통: 0 <= score < success_cut → class 1
    성공: score >= success_cut → class 2

    success_cut은 양수(0 이상) 구간 내에서 상위 success_top_ratio에 해당하는 분위수
    """
    out = df.copy()
    score = safe_numeric(out[score_col])

    positive_scores = score[score >= 0].dropna()
    if len(positive_scores) == 0:
        raise ValueError("양수 score가 없어 3-class 라벨을 만들 수 없습니다.")

    q = 1.0 - success_top_ratio
    success_cut = positive_scores.quantile(q)

    out[label_name] = 1
    out.loc[score < 0, label_name] = 0
    out.loc[score >= success_cut, label_name] = 2

    return out, float(success_cut)


# =========================================================
# 4. 실험 함수
# =========================================================
def run_multiclass_experiment(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    experiment_name: str
):
    usable_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]

    if missing_features:
        print(f"[WARN] Missing features excluded: {missing_features}")

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

    models = {
        "logistic": LogisticRegression(
            max_iter=3000,
            random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "xgb": XGBClassifier(
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
    }

    result_rows = []

    for model_name, model in models.items():
        print("\n" + "-" * 60)
        print(f"[MODEL] {model_name}")

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro")
        weighted_f1 = f1_score(y_test, pred, average="weighted")

        print(classification_report(y_test, pred, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, pred))
        print(f"Accuracy     : {acc:.4f}")
        print(f"Macro F1     : {macro_f1:.4f}")
        print(f"Weighted F1  : {weighted_f1:.4f}")

        result_rows.append({
            "experiment": experiment_name,
            "model": model_name,
            "target": target_col,
            "n_rows": len(df),
            "n_features": len(usable_features),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        })

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(RESULT_DIR / f"{experiment_name}_metrics.csv", index=False, encoding="utf-8-sig")
    return result_df


# =========================================================
# 5. main
# =========================================================
def main():
    hitter = pd.read_csv(HITTER_PATH)
    pitcher = pd.read_csv(PITCHER_PATH)
    pitcher = add_pitcher_role(pitcher)

    print("\n[HITTER COLUMNS]")
    print(hitter.columns.tolist())

    print("\n[PITCHER COLUMNS]")
    print(pitcher.columns.tolist())

    # -------------------------
    # z-score 결과변수 생성
    # hitter: WAR + PA
    # pitcher: WAR + IP
    # -------------------------
    hitter = make_zscore(
        hitter,
        war_col="post_WAR_3y_sum",
        volume_col="post_PA_3y",
        new_col="h_post_score"
    )

    pitcher = make_zscore(
        pitcher,
        war_col="post_WAR_3y_sum",
        volume_col="post_IP_3y",
        new_col="p_post_score"
    )

    # -------------------------
    # feature set
    # -------------------------
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

    ratios = [0.20, 0.30, 0.40]
    results = []

    for r in ratios:
        print("\n" + "#" * 80)
        print(f"[RUN] success_top_ratio = {r:.2f}")
        print("#" * 80)

        # -------------------------
        # 라벨 생성 (원본 복사본 사용)
        # -------------------------
        hitter_r, hitter_cut = make_3class_label(
            hitter.copy(),
            score_col="h_post_score",
            success_top_ratio=r,
            label_name="h_target_3class"
        )

        pitcher_r, pitcher_cut = make_3class_label(
            pitcher.copy(),
            score_col="p_post_score",
            success_top_ratio=r,
            label_name="p_target_3class"
        )

        print("\n" + "-" * 60)
        print(f"[HITTER 3-CLASS] top{int(r*100)}, success_cut={hitter_cut:.4f}")
        print_class_distribution(hitter_r["h_target_3class"], f"h_target_3class_top{int(r*100)}")

        print("\n" + "-" * 60)
        print(f"[PITCHER 3-CLASS] top{int(r*100)}, success_cut={pitcher_cut:.4f}")
        print_class_distribution(pitcher_r["p_target_3class"], f"p_target_3class_top{int(r*100)}")

        # -------------------------
        # Hitter 실험
        # -------------------------
        res_h = run_multiclass_experiment(
            df=hitter_r,
            features=hitter_features,
            target_col="h_target_3class",
            experiment_name=f"hitter_final_3class_top{int(r*100)}"
        )
        res_h["success_top_ratio"] = r
        res_h["success_cut"] = hitter_cut
        results.append(res_h)

        # -------------------------
        # Pitcher 실험
        # -------------------------
        res_p = run_multiclass_experiment(
            df=pitcher_r,
            features=pitcher_features,
            target_col="p_target_3class",
            experiment_name=f"pitcher_final_3class_top{int(r*100)}"
        )
        res_p["success_top_ratio"] = r
        res_p["success_cut"] = pitcher_cut
        results.append(res_p)

    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(
        RESULT_DIR / "all_final_3class_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("[FINAL 3-CLASS SUMMARY]")
    print("=" * 80)
    print(all_results.sort_values(
        ["experiment", "accuracy"],
        ascending=[True, False]
    ))


if __name__ == "__main__":
    main()