# -*- coding: utf-8 -*-
"""
2-class cutoff comparison
- Final 2-class 기준을 뒷받침하기 위한 cutoff 비교 실험
- Hitter: top 30%, 40%, 50% 비교
- Pitcher: top 10%, 20%, 30% 비교
- Composite Score = z(post_WAR_3y_sum) + z(post_playtime_3y)
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")


# =========================================================
# 0. 경로 설정
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compare_2class"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HITTER_PATH = PROCESSED_DIR / "final_hitter_fa_dataset_2011_2024_FAonly_clean_posgroup.csv"
PITCHER_PATH = PROCESSED_DIR / "final_pitcher_fa_dataset_2011_2024_FAonly_clean.csv"


# =========================================================
# 1. 유틸
# =========================================================
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def add_pitcher_role(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Pos" in out.columns:
        out["Pos"] = out["Pos"].fillna("").astype(str).str.upper()
    else:
        out["Pos"] = ""

    out["Role"] = out["Pos"].apply(lambda x: "SP" if "SP" in x else "RP")
    return out


def create_2class_label(
    df: pd.DataFrame,
    post_war_col: str,
    post_playtime_col: str,
    success_top_ratio: float,
    label_col: str
):
    out = df.copy()

    temp = out[[post_war_col, post_playtime_col]].copy()
    for c in temp.columns:
        temp[c] = safe_numeric(temp[c]).fillna(safe_numeric(temp[c]).median())

    scaler = StandardScaler()
    z = scaler.fit_transform(temp)

    out["z_post_WAR"] = z[:, 0]
    out["z_post_playtime"] = z[:, 1]
    out["composite_score"] = out["z_post_WAR"] + out["z_post_playtime"]

    success_cut = out["composite_score"].quantile(1.0 - success_top_ratio)
    out[label_col] = (out["composite_score"] >= success_cut).astype(int)

    return out, float(success_cut)


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ]
    )

    return preprocessor


def find_best_threshold_by_accuracy(y_true, y_proba):
    best_t = 0.50
    best_acc = -1.0

    for t in np.arange(0.10, 0.91, 0.01):
        pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y_true, pred)

        if acc > best_acc:
            best_acc = acc
            best_t = t

    return float(best_t), float(best_acc)


def evaluate_models(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    group_name: str,
    success_top_ratio: float,
    success_cut: float
):
    usable_features = [f for f in features if f in df.columns]

    X = df[usable_features].copy()
    y = df[target_col].copy()

    success_rate = y.mean()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp,
        test_size=0.25,
        stratify=y_temp,
        random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    models = {
        "logistic": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "xgb": XGBClassifier(
            n_estimators=300,
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

    rows = []

    for model_name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        valid_proba = pipe.predict_proba(X_valid)[:, 1]
        best_threshold, valid_acc = find_best_threshold_by_accuracy(
            y_valid,
            valid_proba
        )

        test_proba = pipe.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= best_threshold).astype(int)

        row = {
            "group": group_name,
            "success_top_ratio": success_top_ratio,
            "success_percent": int(success_top_ratio * 100),
            "success_cut": success_cut,
            "model": model_name,
            "n_rows": len(df),
            "n_features": len(usable_features),
            "success_rate": success_rate,
            "best_threshold": best_threshold,
            "accuracy": accuracy_score(y_test, test_pred),
            "roc_auc": roc_auc_score(y_test, test_proba),
            "pr_auc": average_precision_score(y_test, test_proba),
            "precision_success": precision_score(y_test, test_pred, zero_division=0),
            "recall_success": recall_score(y_test, test_pred, zero_division=0),
            "f1_success": f1_score(y_test, test_pred, zero_division=0),
        }

        rows.append(row)

    return rows


# =========================================================
# 2. main
# =========================================================
def main():
    hitter = pd.read_csv(HITTER_PATH)
    pitcher = pd.read_csv(PITCHER_PATH)
    pitcher = add_pitcher_role(pitcher)

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

    # 최종 기준을 중심으로 앞뒤 후보 비교
    hitter_ratios = [0.30, 0.40, 0.50]
    pitcher_ratios = [0.10, 0.20, 0.30]

    all_rows = []

    for r in hitter_ratios:
        hitter_labeled, cut = create_2class_label(
            df=hitter,
            post_war_col="post_WAR_3y_sum",
            post_playtime_col="post_PA_3y",
            success_top_ratio=r,
            label_col="h_success_main"
        )

        print(f"\n[HITTER] top {int(r*100)}%, cut={cut:.4f}")
        print(hitter_labeled["h_success_main"].value_counts(normalize=True))

        rows = evaluate_models(
            df=hitter_labeled,
            features=hitter_features,
            target_col="h_success_main",
            group_name="hitter",
            success_top_ratio=r,
            success_cut=cut
        )
        all_rows.extend(rows)

    for r in pitcher_ratios:
        pitcher_labeled, cut = create_2class_label(
            df=pitcher,
            post_war_col="post_WAR_3y_sum",
            post_playtime_col="post_IP_3y",
            success_top_ratio=r,
            label_col="p_success_main"
        )

        print(f"\n[PITCHER] top {int(r*100)}%, cut={cut:.4f}")
        print(pitcher_labeled["p_success_main"].value_counts(normalize=True))

        rows = evaluate_models(
            df=pitcher_labeled,
            features=pitcher_features,
            target_col="p_success_main",
            group_name="pitcher",
            success_top_ratio=r,
            success_cut=cut
        )
        all_rows.extend(rows)

    result_df = pd.DataFrame(all_rows)

    result_path = OUTPUT_DIR / "compare_2class_cutoffs.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("[2-CLASS CUTOFF COMPARISON]")
    print("=" * 80)
    print(result_df.sort_values(["group", "success_top_ratio", "model"]))

    print(f"\n[OK] Saved: {result_path}")


if __name__ == "__main__":
    main()