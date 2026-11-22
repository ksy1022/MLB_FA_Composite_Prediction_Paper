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
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "project_source" / "review"
OUT = ROOT / "cutoff_revalidation_results"
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


def provisional_label(df, war_col, pt_col, ratio, target):
    if target == "war":
        score = pd.to_numeric(df[war_col], errors="coerce")
        score = score.fillna(score.median())
    else:
        a = df[[war_col, pt_col]].apply(pd.to_numeric, errors="coerce")
        a = a.fillna(a.median())
        score = StandardScaler().fit_transform(a).sum(axis=1)
        score = pd.Series(score, index=df.index)
    return (score >= score.quantile(1 - ratio)).astype(int)


def train_test_labels(train, test, war_col, pt_col, ratio, target):
    if target == "war":
        med = pd.to_numeric(train[war_col], errors="coerce").median()
        tr_score = pd.to_numeric(train[war_col], errors="coerce").fillna(med)
        te_score = pd.to_numeric(test[war_col], errors="coerce").fillna(med)
    else:
        tr = train[[war_col, pt_col]].apply(pd.to_numeric, errors="coerce")
        med = tr.median()
        tr = tr.fillna(med)
        te = test[[war_col, pt_col]].apply(pd.to_numeric, errors="coerce").fillna(med)
        scaler = StandardScaler().fit(tr)
        tr_score = pd.Series(scaler.transform(tr).sum(axis=1), index=train.index)
        te_score = pd.Series(scaler.transform(te).sum(axis=1), index=test.index)
    cutoff = float(tr_score.quantile(1 - ratio))
    return (
        (tr_score >= cutoff).astype(int),
        (te_score >= cutoff).astype(int),
        cutoff,
    )


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


def models(pos_weight):
    return {
        "logistic": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "xgb": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            scale_pos_weight=pos_weight, random_state=42, n_jobs=-1
        ),
    }


def run_role(df, role, features, war_col, pt_col):
    rows = []
    groups = df["player_key"].fillna(df["player_norm"]).astype(str)
    for target in ["composite", "war"]:
        for ratio in [0.20, 0.30, 0.40]:
            provisional = provisional_label(df, war_col, pt_col, ratio, target)
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            for fold, (tri, tei) in enumerate(
                cv.split(df, provisional, groups), start=1
            ):
                train, test = df.iloc[tri].copy(), df.iloc[tei].copy()
                y_train, y_test, cutoff = train_test_labels(
                    train, test, war_col, pt_col, ratio, target
                )
                x_train = train[features]
                x_test = test[features]
                neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
                for name, model in models(neg / max(pos, 1)).items():
                    pipe = Pipeline([
                        ("prep", preprocessor(x_train)),
                        ("model", model),
                    ])
                    pipe.fit(x_train, y_train)
                    prob = pipe.predict_proba(x_test)[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    rows.append({
                        "role": role, "target": target,
                        "top_percent": int(ratio * 100), "fold": fold,
                        "model": name, "n_train": len(train), "n_test": len(test),
                        "train_positive_rate": y_train.mean(),
                        "test_positive_rate": y_test.mean(), "train_cutoff": cutoff,
                        "accuracy": accuracy_score(y_test, pred),
                        "precision": precision_score(y_test, pred, zero_division=0),
                        "recall": recall_score(y_test, pred, zero_division=0),
                        "f1": f1_score(y_test, pred, zero_division=0),
                        "roc_auc": roc_auc_score(y_test, prob),
                        "pr_auc": average_precision_score(y_test, prob),
                    })
    return rows


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
    rows = []
    rows += run_role(
        hitter, "Hitter", HITTER_FEATURES, "post_WAR_3y_sum", "post_PA_3y"
    )
    for role in ["SP", "RP"]:
        part = pitcher[pitcher["Role"] == role].reset_index(drop=True)
        rows += run_role(
            part, role, PITCHER_FEATURES, "post_WAR_3y_sum", "post_IP_3y"
        )
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "fold_results.csv", index=False, encoding="utf-8-sig")
    summary = (
        raw.groupby(["role", "target", "top_percent", "model"])
        .agg(
            n=("fold", "size"),
            f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
            pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
            recall_mean=("recall", "mean"), accuracy_mean=("accuracy", "mean"),
            test_positive_rate=("test_positive_rate", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
