# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.modeling.final_2_class_model import (
    HITTER_PATH,
    load_and_prepare_hitter,
    HITTER_EXPANDED_FEATURES,
    build_preprocessor,
    get_model_and_param_grid,
)

RESULT_DIR = Path("outputs/label_comparison")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(X, y, label_name):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    preprocessor, _, _ = build_preprocessor(X_train)

    model, param_grid = get_model_and_param_grid("xgb", y_train)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    from sklearn.model_selection import GridSearchCV

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    result = {
        "label_type": label_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "success_ratio": y.mean(),
    }

    return result


def main():

    hitter = load_and_prepare_hitter(HITTER_PATH)

    features = [
        f for f in HITTER_EXPANDED_FEATURES
        if f in hitter.columns
    ]

    X = hitter[features].copy()

    # -----------------------------
    # CASE 1: Composite Label
    # -----------------------------
    y_composite = hitter["h_success_main"]

    composite_result = evaluate_model(
        X,
        y_composite,
        "Composite Label"
    )

    # -----------------------------
    # CASE 2: WAR-only Label
    # -----------------------------
    war_threshold = hitter["post_WAR_3y_sum"].quantile(0.6)

    y_war_only = (
        hitter["post_WAR_3y_sum"] >= war_threshold
    ).astype(int)

    war_result = evaluate_model(
        X,
        y_war_only,
        "WAR-only Label"
    )

    results = pd.DataFrame([
        composite_result,
        war_result,
    ])

    print("\n========== LABEL COMPARISON ==========")
    print(results)

    save_path = RESULT_DIR / "label_comparison_results.csv"
    results.to_csv(save_path, index=False)

    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()