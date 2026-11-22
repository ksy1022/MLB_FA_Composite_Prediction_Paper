from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "project_source" / "review"
OUT = ROOT / "cutoff_revalidation_3class_results"
OUT.mkdir(exist_ok=True)

HITTER_FEATURES = [
    "Pos_group", "age_at_contract", "pre_PA_3y", "pre_WAR_3y_avg",
    "pre_WAR_3y_std", "pre_OPSplus_3y_avg", "pre_BB_rate_3y_avg",
    "pre_K_rate_3y_avg", "AAV", "Yrs",
]
PITCHER_FEATURES = [
    "age_at_contract", "pre_IP_3y", "pre_WAR_3y_avg", "pre_WAR_3y_std",
    "pre_ERA_3y_avg", "pre_FIP_3y_avg", "pre_KBB_3y_avg",
    "pre_GS_3y_sum", "pre_G_3y_sum", "AAV", "Yrs",
]
LABELS = [0, 1, 2]
CLASS_NAMES = ["low", "mid", "high"]


def score_pair(train, test, war_col, pt_col, target):
    if target == "war":
        median = pd.to_numeric(train[war_col], errors="coerce").median()
        train_score = pd.to_numeric(train[war_col], errors="coerce").fillna(median)
        test_score = pd.to_numeric(test[war_col], errors="coerce").fillna(median)
    else:
        train_values = train[[war_col, pt_col]].apply(pd.to_numeric, errors="coerce")
        median = train_values.median()
        train_values = train_values.fillna(median)
        test_values = (
            test[[war_col, pt_col]].apply(pd.to_numeric, errors="coerce").fillna(median)
        )
        scaler = StandardScaler().fit(train_values)
        train_score = pd.Series(
            scaler.transform(train_values).sum(axis=1), index=train.index
        )
        test_score = pd.Series(
            scaler.transform(test_values).sum(axis=1), index=test.index
        )
    return train_score, test_score


def labels_from_train(train_score, test_score, ratio):
    nonnegative = train_score[train_score >= 0]
    if nonnegative.empty:
        raise ValueError("Training fold has no non-negative outcome scores.")
    high_cutoff = float(nonnegative.quantile(1 - ratio))

    def make(score):
        label = pd.Series(1, index=score.index, dtype=int)
        label.loc[score < 0] = 0
        label.loc[score >= high_cutoff] = 2
        return label

    return make(train_score), make(test_score), high_cutoff


def provisional_label(df, war_col, pt_col, ratio, target):
    train_score, _ = score_pair(df, df, war_col, pt_col, target)
    label, _, _ = labels_from_train(train_score, train_score, ratio)
    return label


def preprocessor(x):
    nums = x.select_dtypes(include=np.number).columns.tolist()
    cats = x.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), nums),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cats),
    ])


def models():
    return {
        "logistic": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "xgb": XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=300,
            learning_rate=0.05, max_depth=4, subsample=0.9,
            colsample_bytree=0.9, eval_metric="mlogloss",
            random_state=42, n_jobs=-1
        ),
    }


def run_role(df, role, features, war_col, pt_col):
    rows, matrix_rows = [], []
    groups = df["player_key"].fillna(df["player_norm"]).astype(str)
    for target in ["composite", "war"]:
        for ratio in [0.20, 0.30, 0.40]:
            provisional = provisional_label(df, war_col, pt_col, ratio, target)
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            for fold, (train_idx, test_idx) in enumerate(
                cv.split(df, provisional, groups), start=1
            ):
                train = df.iloc[train_idx].copy()
                test = df.iloc[test_idx].copy()
                train_score, test_score = score_pair(
                    train, test, war_col, pt_col, target
                )
                y_train, y_test, high_cutoff = labels_from_train(
                    train_score, test_score, ratio
                )
                x_train, x_test = train[features], test[features]
                for model_name, model in models().items():
                    pipe = Pipeline([
                        ("prep", preprocessor(x_train)),
                        ("model", model),
                    ])
                    if model_name == "xgb":
                        weights = compute_sample_weight("balanced", y_train)
                        pipe.fit(x_train, y_train, model__sample_weight=weights)
                    else:
                        pipe.fit(x_train, y_train)
                    pred = pipe.predict(x_test)
                    precision, recall, class_f1, support = (
                        precision_recall_fscore_support(
                            y_test, pred, labels=LABELS, zero_division=0
                        )
                    )
                    row = {
                        "role": role, "target": target,
                        "top_percent": int(ratio * 100), "fold": fold,
                        "model": model_name, "n_train": len(train),
                        "n_test": len(test), "high_cutoff": high_cutoff,
                        "accuracy": accuracy_score(y_test, pred),
                        "macro_f1": f1_score(
                            y_test, pred, labels=LABELS, average="macro",
                            zero_division=0
                        ),
                        "weighted_f1": f1_score(
                            y_test, pred, labels=LABELS, average="weighted",
                            zero_division=0
                        ),
                    }
                    for i, class_name in enumerate(CLASS_NAMES):
                        row[f"{class_name}_precision"] = precision[i]
                        row[f"{class_name}_recall"] = recall[i]
                        row[f"{class_name}_f1"] = class_f1[i]
                        row[f"{class_name}_support"] = support[i]
                        row[f"train_{class_name}_rate"] = (y_train == i).mean()
                        row[f"test_{class_name}_rate"] = (y_test == i).mean()
                    rows.append(row)

                    matrix = confusion_matrix(y_test, pred, labels=LABELS)
                    for true_class in LABELS:
                        for pred_class in LABELS:
                            matrix_rows.append({
                                "role": role, "target": target,
                                "top_percent": int(ratio * 100), "fold": fold,
                                "model": model_name, "true_class": true_class,
                                "pred_class": pred_class,
                                "count": int(matrix[true_class, pred_class]),
                            })
    return rows, matrix_rows


def main():
    hitter = pd.read_csv(
        DATA / "final_hitter_fa_dataset_2011_2024_FAonly_clean_posgroup.csv"
    )
    pitcher = pd.read_csv(
        DATA / "final_pitcher_fa_dataset_2011_2024_FAonly_clean.csv"
    )
    pitcher["Role"] = np.where(
        pitcher["Pos"].fillna("").str.upper().str.contains("SP"), "SP", "RP"
    )

    rows, matrices = [], []
    role_rows, role_matrices = run_role(
        hitter, "Hitter", HITTER_FEATURES, "post_WAR_3y_sum", "post_PA_3y"
    )
    rows += role_rows
    matrices += role_matrices
    for role in ["SP", "RP"]:
        part = pitcher[pitcher["Role"] == role].reset_index(drop=True)
        role_rows, role_matrices = run_role(
            part, role, PITCHER_FEATURES, "post_WAR_3y_sum", "post_IP_3y"
        )
        rows += role_rows
        matrices += role_matrices

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "fold_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(matrices).to_csv(
        OUT / "confusion_matrices.csv", index=False, encoding="utf-8-sig"
    )

    metric_columns = [
        "accuracy", "macro_f1", "weighted_f1",
        "low_recall", "mid_recall", "high_recall",
        "low_f1", "mid_f1", "high_f1",
        "test_low_rate", "test_mid_rate", "test_high_rate",
    ]
    summary = (
        raw.groupby(["role", "target", "top_percent", "model"])[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")

    wide = summary.pivot(
        index=["role", "top_percent", "model"],
        columns="target",
        values=["macro_f1_mean", "high_f1_mean", "accuracy_mean"],
    )
    differences = pd.DataFrame(index=wide.index)
    for metric in ["macro_f1_mean", "high_f1_mean", "accuracy_mean"]:
        differences[f"{metric}_composite_minus_war"] = (
            wide[(metric, "composite")] - wide[(metric, "war")]
        )
    differences.reset_index().to_csv(
        OUT / "composite_minus_war.csv", index=False, encoding="utf-8-sig"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
