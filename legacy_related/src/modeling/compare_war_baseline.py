# -*- coding: utf-8 -*-
"""
Composite Score vs WAR-only Baseline
Role-specific Actual Success Comparison

핵심 수정 사항
---------------------------------------------------------
1. 타자
   - Composite Score 상위 40%
   - WAR 상위 40%

2. 선발투수(SP)
   - SP 내부 Composite Score 상위 20%
   - SP 내부 WAR 상위 20%

3. 불펜투수(RP)
   - RP 내부 Composite Score 상위 20%
   - RP 내부 WAR 상위 20%

4. 투수 Composite Score의 z-score도 SP와 RP별로 계산

5. 두 라벨을 학습한 모델의 예측을
   동일한 actual_success 기준과 비교

Actual success operational benchmark
---------------------------------------------------------
Hitter:
    post_WAR_3y_sum >= 0.4
    AND post_PA_3y >= 622.0

Starting Pitcher:
    post_WAR_3y_sum >= 1.3
    AND post_IP_3y >= 280.6

Relief Pitcher:
    post_WAR_3y_sum >= 0.5
    AND post_IP_3y >= 94.2

위 수치는 현재 분석 데이터의 그룹별 중앙값이다.

실행 위치
---------------------------------------------------------
프로젝트 루트

실행 명령
---------------------------------------------------------
python src/experiments/compare_composite_war_actual_success_rolewise.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =========================================================
# 0. 프로젝트 경로
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

# 파일 위치:
# PROJECT_ROOT/src/experiments/현재파일.py
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 기존 2-class 모델 코드의 전처리·모델 설정 재사용
from src.modeling.final_2_class_model import (  # noqa: E402
    HITTER_PATH,
    PITCHER_PATH,
    HITTER_EXPANDED_FEATURES,
    PITCHER_SP_EXPANDED_FEATURES,
    PITCHER_RP_EXPANDED_FEATURES,
    build_preprocessor,
    engineer_hitter_features,
    engineer_pitcher_features,
    find_best_threshold_by_accuracy,
    get_model_and_param_grid,
)


# =========================================================
# 1. 설정
# =========================================================

RANDOM_STATE = 42

RESULT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "composite_war_actual_success_rolewise"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_NAMES = [
    "logistic",
    "rf",
    "xgb",
]


# ---------------------------------------------------------
# 성공 상위 비율
# ---------------------------------------------------------

HITTER_SUCCESS_TOP_RATIO = 0.40
PITCHER_SUCCESS_TOP_RATIO = 0.20


# ---------------------------------------------------------
# Actual success 기준
# 현재 데이터의 그룹별 중앙값
# ---------------------------------------------------------

HITTER_ACTUAL_WAR_MIN = 0.4
HITTER_ACTUAL_PA_MIN = 622.0

SP_ACTUAL_WAR_MIN = 1.3
SP_ACTUAL_IP_MIN = 280.6

RP_ACTUAL_WAR_MIN = 0.5
RP_ACTUAL_IP_MIN = 94.2


# =========================================================
# 2. 공통 유틸 함수
# =========================================================

def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """Series를 숫자형으로 안전하게 변환한다."""
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    context: str,
) -> None:
    """필수 컬럼이 존재하는지 확인한다."""
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{context}에 필요한 컬럼이 없습니다: {missing}"
        )


def fill_numeric_median(
    series: pd.Series,
) -> pd.Series:
    """숫자형으로 변환한 뒤 중앙값으로 결측치를 대체한다."""
    numeric = safe_numeric(series)

    median_value = numeric.median()

    return numeric.fillna(median_value)


def calculate_binary_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray,
) -> dict[str, float]:
    """이진 분류 평가 지표를 계산한다."""
    y_true_array = np.asarray(y_true).astype(int)
    y_pred_array = np.asarray(y_pred).astype(int)
    y_probability_array = np.asarray(
        y_probability
    ).astype(float)

    result = {
        "accuracy": accuracy_score(
            y_true_array,
            y_pred_array,
        ),
        "precision": precision_score(
            y_true_array,
            y_pred_array,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true_array,
            y_pred_array,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true_array,
            y_pred_array,
            zero_division=0,
        ),
        "pr_auc": average_precision_score(
            y_true_array,
            y_probability_array,
        ),
    }

    if len(np.unique(y_true_array)) == 2:
        result["roc_auc"] = roc_auc_score(
            y_true_array,
            y_probability_array,
        )
    else:
        result["roc_auc"] = np.nan

    return result


# =========================================================
# 3. 타자 Composite / WAR 라벨 생성
# =========================================================

def prepare_hitter_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    타자 라벨 생성

    Composite:
        z(post WAR) + z(post PA)
        상위 40%

    WAR-only:
        post WAR 상위 40%
    """
    out = df.copy()

    require_columns(
        out,
        [
            "post_WAR_3y_sum",
            "post_PA_3y",
        ],
        "타자 라벨 생성",
    )

    post_war = fill_numeric_median(
        out["post_WAR_3y_sum"]
    )

    post_pa = fill_numeric_median(
        out["post_PA_3y"]
    )

    scaler = StandardScaler()

    standardized = scaler.fit_transform(
        pd.DataFrame(
            {
                "post_war": post_war,
                "post_pa": post_pa,
            }
        )
    )

    out["h_z_post_WAR"] = standardized[:, 0]
    out["h_z_post_playtime"] = standardized[:, 1]

    out["h_composite_score_rolewise"] = (
        out["h_z_post_WAR"]
        + out["h_z_post_playtime"]
    )

    composite_cutoff = out[
        "h_composite_score_rolewise"
    ].quantile(
        1.0 - HITTER_SUCCESS_TOP_RATIO
    )

    war_cutoff = post_war.quantile(
        1.0 - HITTER_SUCCESS_TOP_RATIO
    )

    out["h_composite_success_rolewise"] = (
        out["h_composite_score_rolewise"]
        >= composite_cutoff
    ).astype(int)

    out["h_war_success_rolewise"] = (
        post_war >= war_cutoff
    ).astype(int)

    cutoff_info = {
        "composite_cutoff": float(
            composite_cutoff
        ),
        "war_cutoff": float(
            war_cutoff
        ),
    }

    print("\n" + "=" * 80)
    print("HITTER ROLE-WISE LABELS")
    print("=" * 80)

    print(
        f"Composite cutoff: "
        f"{composite_cutoff:.4f}"
    )

    print(
        f"WAR-only cutoff: "
        f"{war_cutoff:.4f}"
    )

    print(
        "Composite distribution:",
        out[
            "h_composite_success_rolewise"
        ].value_counts().sort_index().to_dict(),
    )

    print(
        "WAR-only distribution:",
        out[
            "h_war_success_rolewise"
        ].value_counts().sort_index().to_dict(),
    )

    return out, cutoff_info


# =========================================================
# 4. 투수 역할별 Composite / WAR 라벨 생성
# =========================================================

def prepare_pitcher_role_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    SP와 RP를 각각 분리한 뒤 라벨 생성

    각 역할 내부에서:
    - z(post WAR)
    - z(post IP)
    - Composite Score
    - Composite 상위 20%
    - WAR 상위 20%

    기존 문제:
        전체 투수 상위 20%를 먼저 만들면
        RP 성공 비율이 지나치게 낮아질 수 있음.

    수정:
        SP 내부 상위 20%
        RP 내부 상위 20%
    """
    out = df.copy()

    require_columns(
        out,
        [
            "Role",
            "post_WAR_3y_sum",
            "post_IP_3y",
        ],
        "투수 역할별 라벨 생성",
    )

    role_frames: list[pd.DataFrame] = []

    cutoff_rows: list[dict[str, Any]] = []

    for role in ["SP", "RP"]:
        role_df = out[
            out["Role"] == role
        ].copy()

        if role_df.empty:
            raise ValueError(
                f"{role} 데이터가 없습니다."
            )

        post_war = fill_numeric_median(
            role_df["post_WAR_3y_sum"]
        )

        post_ip = fill_numeric_median(
            role_df["post_IP_3y"]
        )

        scaler = StandardScaler()

        standardized = scaler.fit_transform(
            pd.DataFrame(
                {
                    "post_war": post_war,
                    "post_ip": post_ip,
                },
                index=role_df.index,
            )
        )

        role_df[
            "p_role_z_post_WAR"
        ] = standardized[:, 0]

        role_df[
            "p_role_z_post_playtime"
        ] = standardized[:, 1]

        role_df[
            "p_composite_score_rolewise"
        ] = (
            role_df["p_role_z_post_WAR"]
            + role_df["p_role_z_post_playtime"]
        )

        composite_cutoff = role_df[
            "p_composite_score_rolewise"
        ].quantile(
            1.0 - PITCHER_SUCCESS_TOP_RATIO
        )

        war_cutoff = post_war.quantile(
            1.0 - PITCHER_SUCCESS_TOP_RATIO
        )

        role_df[
            "p_composite_success_rolewise"
        ] = (
            role_df["p_composite_score_rolewise"]
            >= composite_cutoff
        ).astype(int)

        role_df[
            "p_war_success_rolewise"
        ] = (
            post_war >= war_cutoff
        ).astype(int)

        role_frames.append(role_df)

        cutoff_rows.append(
            {
                "dataset": f"pitcher_{role.lower()}",
                "role": role,
                "n_players": len(role_df),
                "composite_cutoff": float(
                    composite_cutoff
                ),
                "war_cutoff": float(
                    war_cutoff
                ),
                "composite_success_count": int(
                    role_df[
                        "p_composite_success_rolewise"
                    ].sum()
                ),
                "war_success_count": int(
                    role_df[
                        "p_war_success_rolewise"
                    ].sum()
                ),
                "composite_success_rate": float(
                    role_df[
                        "p_composite_success_rolewise"
                    ].mean()
                ),
                "war_success_rate": float(
                    role_df[
                        "p_war_success_rolewise"
                    ].mean()
                ),
            }
        )

        print("\n" + "=" * 80)
        print(f"{role} ROLE-WISE LABELS")
        print("=" * 80)

        print(
            f"Number of players: {len(role_df)}"
        )

        print(
            f"Composite cutoff: "
            f"{composite_cutoff:.4f}"
        )

        print(
            f"WAR-only cutoff: "
            f"{war_cutoff:.4f}"
        )

        print(
            "Composite distribution:",
            role_df[
                "p_composite_success_rolewise"
            ].value_counts().sort_index().to_dict(),
        )

        print(
            "WAR-only distribution:",
            role_df[
                "p_war_success_rolewise"
            ].value_counts().sort_index().to_dict(),
        )

    combined = pd.concat(
        role_frames,
        axis=0,
    ).sort_index()

    cutoff_df = pd.DataFrame(
        cutoff_rows
    )

    return combined, cutoff_df


# =========================================================
# 5. Actual success 생성
# =========================================================

def add_hitter_actual_success(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    타자 actual success:

    post WAR >= 0.4
    AND post PA >= 622.0
    """
    out = df.copy()

    post_war = safe_numeric(
        out["post_WAR_3y_sum"]
    )

    post_pa = safe_numeric(
        out["post_PA_3y"]
    )

    out["h_actual_success"] = (
        (post_war >= HITTER_ACTUAL_WAR_MIN)
        & (post_pa >= HITTER_ACTUAL_PA_MIN)
    ).astype(int)

    print("\n[HITTER ACTUAL SUCCESS]")

    print(
        f"WAR >= {HITTER_ACTUAL_WAR_MIN}, "
        f"PA >= {HITTER_ACTUAL_PA_MIN}"
    )

    print(
        f"Success: "
        f"{out['h_actual_success'].sum()} "
        f"/ {len(out)} "
        f"({out['h_actual_success'].mean():.4f})"
    )

    return out


def add_pitcher_actual_success(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    역할별 actual success

    SP:
        WAR >= 1.3
        IP >= 280.6

    RP:
        WAR >= 0.5
        IP >= 94.2
    """
    out = df.copy()

    post_war = safe_numeric(
        out["post_WAR_3y_sum"]
    )

    post_ip = safe_numeric(
        out["post_IP_3y"]
    )

    is_sp = out["Role"].eq("SP")
    is_rp = out["Role"].eq("RP")

    sp_success = (
        is_sp
        & (post_war >= SP_ACTUAL_WAR_MIN)
        & (post_ip >= SP_ACTUAL_IP_MIN)
    )

    rp_success = (
        is_rp
        & (post_war >= RP_ACTUAL_WAR_MIN)
        & (post_ip >= RP_ACTUAL_IP_MIN)
    )

    out["p_actual_success"] = (
        sp_success | rp_success
    ).astype(int)

    print("\n[PITCHER ACTUAL SUCCESS]")

    for role in ["SP", "RP"]:
        role_df = out[
            out["Role"] == role
        ]

        print(
            f"{role}: "
            f"{role_df['p_actual_success'].sum()} "
            f"/ {len(role_df)} "
            f"({role_df['p_actual_success'].mean():.4f})"
        )

    return out


# =========================================================
# 6. 동일한 데이터 분할
# =========================================================

def create_shared_split(
    df: pd.DataFrame,
    actual_success_col: str,
) -> dict[str, np.ndarray]:
    """
    Composite 모델과 WAR-only 모델에
    완전히 동일한 선수 분할을 사용한다.

    actual_success 기준으로 층화 분할:
    - Train: 60%
    - Validation: 20%
    - Test: 20%
    """
    actual_y = (
        df[actual_success_col]
        .astype(int)
        .to_numpy()
    )

    indices = np.arange(len(df))

    train_valid_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=actual_y,
        random_state=RANDOM_STATE,
    )

    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=0.25,
        stratify=actual_y[train_valid_idx],
        random_state=RANDOM_STATE,
    )

    return {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
    }


# =========================================================
# 7. 모델 학습 및 평가
# =========================================================

def train_and_evaluate(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    actual_success_col: str,
    split_indices: dict[str, np.ndarray],
    dataset_name: str,
    label_type: str,
    model_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    모델은 Composite 또는 WAR-only 라벨을 학습한다.

    평가:
    1. 학습 대상 라벨 자체에 대한 성능
    2. 공통 actual success에 대한 성능
    """
    usable_features = [
        feature
        for feature in features
        if feature in df.columns
    ]

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        print(
            f"\n[WARNING] {dataset_name} 누락 변수:"
        )

        print(missing_features)

    if not usable_features:
        raise ValueError(
            f"{dataset_name}에서 사용할 feature가 없습니다."
        )

    X = df[
        usable_features
    ].copy()

    y_target = df[
        target_col
    ].astype(int)

    y_actual = df[
        actual_success_col
    ].astype(int)

    train_idx = split_indices["train"]
    valid_idx = split_indices["valid"]
    test_idx = split_indices["test"]

    X_train = X.iloc[
        train_idx
    ].copy()

    X_valid = X.iloc[
        valid_idx
    ].copy()

    X_test = X.iloc[
        test_idx
    ].copy()

    y_train = y_target.iloc[
        train_idx
    ].copy()

    y_valid = y_target.iloc[
        valid_idx
    ].copy()

    y_test_target = y_target.iloc[
        test_idx
    ].copy()

    y_test_actual = y_actual.iloc[
        test_idx
    ].copy()

    print("\n" + "=" * 80)

    print(
        f"{dataset_name} | "
        f"{label_type} | "
        f"{model_name}"
    )

    print("=" * 80)

    print(
        f"Train={len(train_idx)}, "
        f"Valid={len(valid_idx)}, "
        f"Test={len(test_idx)}"
    )

    print(
        "Training target distribution:",
        y_train.value_counts().sort_index().to_dict(),
    )

    preprocessor, _, _ = build_preprocessor(
        X_train
    )

    model, param_grid = get_model_and_param_grid(
        model_name,
        y_train,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    best_model = grid_search.best_estimator_

    valid_probability = (
        best_model.predict_proba(
            X_valid
        )[:, 1]
    )

    best_threshold, validation_accuracy = (
        find_best_threshold_by_accuracy(
            y_valid,
            valid_probability,
        )
    )

    test_probability = (
        best_model.predict_proba(
            X_test
        )[:, 1]
    )

    test_prediction = (
        test_probability >= best_threshold
    ).astype(int)

    target_metrics = calculate_binary_metrics(
        y_true=y_test_target,
        y_pred=test_prediction,
        y_probability=test_probability,
    )

    actual_metrics = calculate_binary_metrics(
        y_true=y_test_actual,
        y_pred=test_prediction,
        y_probability=test_probability,
    )

    actual_confusion = confusion_matrix(
        y_test_actual,
        test_prediction,
        labels=[0, 1],
    )

    true_negative = int(
        actual_confusion[0, 0]
    )

    false_positive = int(
        actual_confusion[0, 1]
    )

    false_negative = int(
        actual_confusion[1, 0]
    )

    true_positive = int(
        actual_confusion[1, 1]
    )

    print(
        f"Best parameters: "
        f"{grid_search.best_params_}"
    )

    print(
        f"Best threshold: "
        f"{best_threshold:.2f}"
    )

    print("\n[TARGET LABEL PERFORMANCE]")

    print(
        f"Accuracy={target_metrics['accuracy']:.4f}, "
        f"Precision={target_metrics['precision']:.4f}, "
        f"Recall={target_metrics['recall']:.4f}, "
        f"F1={target_metrics['f1']:.4f}, "
        f"PR-AUC={target_metrics['pr_auc']:.4f}"
    )

    print("\n[ACTUAL SUCCESS PERFORMANCE]")

    print(
        f"Accuracy={actual_metrics['accuracy']:.4f}, "
        f"Precision={actual_metrics['precision']:.4f}, "
        f"Recall={actual_metrics['recall']:.4f}, "
        f"F1={actual_metrics['f1']:.4f}, "
        f"PR-AUC={actual_metrics['pr_auc']:.4f}"
    )

    print("Actual success confusion matrix:")

    print(actual_confusion)

    metric_row: dict[str, Any] = {
        "dataset": dataset_name,
        "label_type": label_type,
        "model": model_name,
        "target_col": target_col,
        "actual_success_col": actual_success_col,
        "n_total": len(df),
        "train_size": len(train_idx),
        "valid_size": len(valid_idx),
        "test_size": len(test_idx),
        "n_features": len(usable_features),
        "best_threshold": best_threshold,
        "validation_accuracy": validation_accuracy,

        "target_accuracy": target_metrics[
            "accuracy"
        ],
        "target_roc_auc": target_metrics[
            "roc_auc"
        ],
        "target_pr_auc": target_metrics[
            "pr_auc"
        ],
        "target_precision": target_metrics[
            "precision"
        ],
        "target_recall": target_metrics[
            "recall"
        ],
        "target_f1": target_metrics[
            "f1"
        ],

        "actual_accuracy": actual_metrics[
            "accuracy"
        ],
        "actual_roc_auc": actual_metrics[
            "roc_auc"
        ],
        "actual_pr_auc": actual_metrics[
            "pr_auc"
        ],
        "actual_precision": actual_metrics[
            "precision"
        ],
        "actual_recall": actual_metrics[
            "recall"
        ],
        "actual_f1": actual_metrics[
            "f1"
        ],

        "true_negative_count": true_negative,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "true_positive_count": true_positive,

        "predicted_success_count": int(
            test_prediction.sum()
        ),
        "actual_success_count": int(
            y_test_actual.sum()
        ),

        "best_params": str(
            grid_search.best_params_
        ),
    }

    # -----------------------------------------------------
    # 선수별 결과
    # -----------------------------------------------------

    player_result = df.iloc[
        test_idx
    ].copy()

    result_columns = [
        column
        for column in [
            "player_key",
            "player",
            "Name",
            "fa_year",
            "Role",
            "Pos_group",
            "post_WAR_3y_sum",
            "post_PA_3y",
            "post_IP_3y",
            "h_composite_score_rolewise",
            "p_composite_score_rolewise",
            target_col,
            actual_success_col,
        ]
        if column in player_result.columns
    ]

    player_result = player_result[
        result_columns
    ].copy()

    player_result["dataset"] = dataset_name
    player_result["label_type"] = label_type
    player_result["model"] = model_name

    player_result[
        "target_label"
    ] = y_test_target.to_numpy()

    player_result[
        "actual_success"
    ] = y_test_actual.to_numpy()

    player_result[
        "predicted_success"
    ] = test_prediction

    player_result[
        "predicted_probability"
    ] = test_probability

    player_result[
        "correct_against_target"
    ] = (
        player_result["predicted_success"]
        == player_result["target_label"]
    )

    player_result[
        "correct_against_actual"
    ] = (
        player_result["predicted_success"]
        == player_result["actual_success"]
    )

    player_result[
        "true_positive"
    ] = (
        (player_result["predicted_success"] == 1)
        & (player_result["actual_success"] == 1)
    )

    player_result[
        "false_positive"
    ] = (
        (player_result["predicted_success"] == 1)
        & (player_result["actual_success"] == 0)
    )

    player_result[
        "false_negative"
    ] = (
        (player_result["predicted_success"] == 0)
        & (player_result["actual_success"] == 1)
    )

    return metric_row, player_result


# =========================================================
# 8. 그룹별 Composite vs WAR 실행
# =========================================================

def run_group_comparison(
    df: pd.DataFrame,
    features: list[str],
    composite_target: str,
    war_target: str,
    actual_success_col: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    하나의 그룹에서 동일한 데이터 분할로:

    Composite 라벨 모델
    WAR-only 라벨 모델

    각각 Logistic, RF, XGBoost 실행
    """
    split_indices = create_shared_split(
        df=df,
        actual_success_col=actual_success_col,
    )

    metric_rows: list[dict[str, Any]] = []

    player_frames: list[pd.DataFrame] = []

    target_settings = [
        (
            "composite",
            composite_target,
        ),
        (
            "war_only",
            war_target,
        ),
    ]

    for label_type, target_col in target_settings:
        for model_name in MODEL_NAMES:
            metrics, player_result = (
                train_and_evaluate(
                    df=df,
                    features=features,
                    target_col=target_col,
                    actual_success_col=(
                        actual_success_col
                    ),
                    split_indices=split_indices,
                    dataset_name=dataset_name,
                    label_type=label_type,
                    model_name=model_name,
                )
            )

            metric_rows.append(metrics)
            player_frames.append(player_result)

    return (
        pd.DataFrame(metric_rows),
        pd.concat(
            player_frames,
            ignore_index=True,
        ),
    )


# =========================================================
# 9. 라벨 자체의 actual success 일치도
# =========================================================

def evaluate_label_definition(
    df: pd.DataFrame,
    composite_target: str,
    war_target: str,
    actual_success_col: str,
    dataset_name: str,
) -> pd.DataFrame:
    """
    모델 학습 이전에 라벨 정의 자체가
    actual success와 얼마나 일치하는지 비교한다.
    """
    rows: list[dict[str, Any]] = []

    for label_type, label_col in [
        ("composite", composite_target),
        ("war_only", war_target),
    ]:
        y_actual = df[
            actual_success_col
        ].astype(int)

        y_label = df[
            label_col
        ].astype(int)

        # 라벨 자체에는 별도 probability가 없으므로
        # 0/1 라벨을 score처럼 사용
        metrics = calculate_binary_metrics(
            y_true=y_actual,
            y_pred=y_label,
            y_probability=y_label,
        )

        confusion = confusion_matrix(
            y_actual,
            y_label,
            labels=[0, 1],
        )

        rows.append(
            {
                "dataset": dataset_name,
                "label_type": label_type,
                "label_col": label_col,
                "n_players": len(df),
                "label_success_count": int(
                    y_label.sum()
                ),
                "actual_success_count": int(
                    y_actual.sum()
                ),
                "agreement_rate": float(
                    (y_label == y_actual).mean()
                ),
                "accuracy": metrics[
                    "accuracy"
                ],
                "precision": metrics[
                    "precision"
                ],
                "recall": metrics[
                    "recall"
                ],
                "f1": metrics[
                    "f1"
                ],
                "true_negative": int(
                    confusion[0, 0]
                ),
                "false_positive": int(
                    confusion[0, 1]
                ),
                "false_negative": int(
                    confusion[1, 0]
                ),
                "true_positive": int(
                    confusion[1, 1]
                ),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# 10. 선수별 라벨 비교표
# =========================================================

def create_player_label_comparison(
    df: pd.DataFrame,
    dataset_name: str,
    composite_target: str,
    war_target: str,
    actual_success_col: str,
    composite_score_col: str,
    playtime_col: str,
) -> pd.DataFrame:
    columns = [
        column
        for column in [
            "player_key",
            "player",
            "Name",
            "fa_year",
            "Role",
            "Pos_group",
            "post_WAR_3y_sum",
            playtime_col,
            composite_score_col,
            composite_target,
            war_target,
            actual_success_col,
        ]
        if column in df.columns
    ]

    result = df[
        columns
    ].copy()

    result["dataset"] = dataset_name

    result[
        "composite_war_agreement"
    ] = (
        result[composite_target]
        == result[war_target]
    )

    result[
        "composite_only_success"
    ] = (
        (result[composite_target] == 1)
        & (result[war_target] == 0)
    )

    result[
        "war_only_success"
    ] = (
        (result[composite_target] == 0)
        & (result[war_target] == 1)
    )

    result[
        "composite_matches_actual"
    ] = (
        result[composite_target]
        == result[actual_success_col]
    )

    result[
        "war_matches_actual"
    ] = (
        result[war_target]
        == result[actual_success_col]
    )

    result[
        "composite_better_case"
    ] = (
        result["composite_matches_actual"]
        & ~result["war_matches_actual"]
    )

    result[
        "war_better_case"
    ] = (
        ~result["composite_matches_actual"]
        & result["war_matches_actual"]
    )

    return result


# =========================================================
# 11. 최종 논문용 비교표
# =========================================================

def create_final_model_summary(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    selected_columns = [
        "dataset",
        "label_type",
        "model",
        "actual_accuracy",
        "actual_roc_auc",
        "actual_pr_auc",
        "actual_precision",
        "actual_recall",
        "actual_f1",
        "predicted_success_count",
        "actual_success_count",
        "true_positive_count",
        "false_positive_count",
        "false_negative_count",
    ]

    return (
        metrics_df[
            selected_columns
        ]
        .sort_values(
            by=[
                "dataset",
                "model",
                "label_type",
            ]
        )
        .reset_index(drop=True)
    )


def create_pairwise_difference_table(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Composite - WAR-only 성능 차이를 계산한다.

    양수:
        Composite가 우수

    음수:
        WAR-only가 우수
    """
    metric_columns = [
        "actual_accuracy",
        "actual_roc_auc",
        "actual_pr_auc",
        "actual_precision",
        "actual_recall",
        "actual_f1",
    ]

    composite = summary_df[
        summary_df["label_type"] == "composite"
    ].copy()

    war = summary_df[
        summary_df["label_type"] == "war_only"
    ].copy()

    merged = composite.merge(
        war,
        on=[
            "dataset",
            "model",
        ],
        suffixes=(
            "_composite",
            "_war",
        ),
    )

    result = merged[
        [
            "dataset",
            "model",
        ]
    ].copy()

    for metric in metric_columns:
        result[
            f"{metric}_composite"
        ] = merged[
            f"{metric}_composite"
        ]

        result[
            f"{metric}_war"
        ] = merged[
            f"{metric}_war"
        ]

        result[
            f"{metric}_difference"
        ] = (
            merged[
                f"{metric}_composite"
            ]
            - merged[
                f"{metric}_war"
            ]
        )

    return result


# =========================================================
# 12. Main
# =========================================================

def main() -> None:
    print("=" * 80)

    print(
        "ROLE-SPECIFIC COMPOSITE VS WAR-ONLY "
        "ACTUAL SUCCESS COMPARISON"
    )

    print("=" * 80)

    hitter_path = PROJECT_ROOT / HITTER_PATH
    pitcher_path = PROJECT_ROOT / PITCHER_PATH

    if not hitter_path.exists():
        raise FileNotFoundError(
            f"타자 데이터 파일이 없습니다: "
            f"{hitter_path}"
        )

    if not pitcher_path.exists():
        raise FileNotFoundError(
            f"투수 데이터 파일이 없습니다: "
            f"{pitcher_path}"
        )

    # -----------------------------------------------------
    # 원본 데이터 로드 및 기존 feature engineering
    # -----------------------------------------------------

    hitter = pd.read_csv(
        hitter_path
    )

    pitcher = pd.read_csv(
        pitcher_path
    )

    hitter = engineer_hitter_features(
        hitter
    )

    pitcher = engineer_pitcher_features(
        pitcher
    )

    # -----------------------------------------------------
    # 새 비교용 라벨 생성
    # -----------------------------------------------------

    hitter, hitter_cutoffs = (
        prepare_hitter_labels(
            hitter
        )
    )

    pitcher, pitcher_cutoffs = (
        prepare_pitcher_role_labels(
            pitcher
        )
    )

    # -----------------------------------------------------
    # Actual success 생성
    # -----------------------------------------------------

    hitter = add_hitter_actual_success(
        hitter
    )

    pitcher = add_pitcher_actual_success(
        pitcher
    )

    pitcher_sp = pitcher[
        pitcher["Role"] == "SP"
    ].copy()

    pitcher_rp = pitcher[
        pitcher["Role"] == "RP"
    ].copy()

    print("\n[DATA SIZE]")

    print(f"Hitter: {len(hitter)}")

    print(
        f"Starting Pitcher: "
        f"{len(pitcher_sp)}"
    )

    print(
        f"Relief Pitcher: "
        f"{len(pitcher_rp)}"
    )

    # -----------------------------------------------------
    # Cutoff 저장
    # -----------------------------------------------------

    cutoff_rows = [
        {
            "dataset": "hitter",
            "role": "Hitter",
            "n_players": len(hitter),
            "success_top_ratio": (
                HITTER_SUCCESS_TOP_RATIO
            ),
            "composite_cutoff": (
                hitter_cutoffs[
                    "composite_cutoff"
                ]
            ),
            "war_cutoff": (
                hitter_cutoffs[
                    "war_cutoff"
                ]
            ),
            "actual_war_min": (
                HITTER_ACTUAL_WAR_MIN
            ),
            "actual_playtime_min": (
                HITTER_ACTUAL_PA_MIN
            ),
            "playtime_unit": "PA",
        }
    ]

    for _, row in pitcher_cutoffs.iterrows():
        if row["role"] == "SP":
            actual_war_min = (
                SP_ACTUAL_WAR_MIN
            )

            actual_playtime_min = (
                SP_ACTUAL_IP_MIN
            )
        else:
            actual_war_min = (
                RP_ACTUAL_WAR_MIN
            )

            actual_playtime_min = (
                RP_ACTUAL_IP_MIN
            )

        cutoff_rows.append(
            {
                "dataset": row["dataset"],
                "role": row["role"],
                "n_players": row["n_players"],
                "success_top_ratio": (
                    PITCHER_SUCCESS_TOP_RATIO
                ),
                "composite_cutoff": (
                    row["composite_cutoff"]
                ),
                "war_cutoff": (
                    row["war_cutoff"]
                ),
                "actual_war_min": (
                    actual_war_min
                ),
                "actual_playtime_min": (
                    actual_playtime_min
                ),
                "playtime_unit": "IP",
            }
        )

    cutoff_df = pd.DataFrame(
        cutoff_rows
    )

    cutoff_df.to_csv(
        RESULT_DIR
        / "role_specific_success_criteria.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 라벨 정의 자체와 actual success 비교
    # -----------------------------------------------------

    label_definition_frames = [
        evaluate_label_definition(
            df=hitter,
            composite_target=(
                "h_composite_success_rolewise"
            ),
            war_target=(
                "h_war_success_rolewise"
            ),
            actual_success_col=(
                "h_actual_success"
            ),
            dataset_name="hitter",
        ),
        evaluate_label_definition(
            df=pitcher_sp,
            composite_target=(
                "p_composite_success_rolewise"
            ),
            war_target=(
                "p_war_success_rolewise"
            ),
            actual_success_col=(
                "p_actual_success"
            ),
            dataset_name="pitcher_sp",
        ),
        evaluate_label_definition(
            df=pitcher_rp,
            composite_target=(
                "p_composite_success_rolewise"
            ),
            war_target=(
                "p_war_success_rolewise"
            ),
            actual_success_col=(
                "p_actual_success"
            ),
            dataset_name="pitcher_rp",
        ),
    ]

    label_definition_summary = pd.concat(
        label_definition_frames,
        ignore_index=True,
    )

    label_definition_summary.to_csv(
        RESULT_DIR
        / "label_definition_vs_actual_success.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 선수별 라벨 비교
    # -----------------------------------------------------

    hitter_label_cases = (
        create_player_label_comparison(
            df=hitter,
            dataset_name="hitter",
            composite_target=(
                "h_composite_success_rolewise"
            ),
            war_target=(
                "h_war_success_rolewise"
            ),
            actual_success_col=(
                "h_actual_success"
            ),
            composite_score_col=(
                "h_composite_score_rolewise"
            ),
            playtime_col="post_PA_3y",
        )
    )

    sp_label_cases = (
        create_player_label_comparison(
            df=pitcher_sp,
            dataset_name="pitcher_sp",
            composite_target=(
                "p_composite_success_rolewise"
            ),
            war_target=(
                "p_war_success_rolewise"
            ),
            actual_success_col=(
                "p_actual_success"
            ),
            composite_score_col=(
                "p_composite_score_rolewise"
            ),
            playtime_col="post_IP_3y",
        )
    )

    rp_label_cases = (
        create_player_label_comparison(
            df=pitcher_rp,
            dataset_name="pitcher_rp",
            composite_target=(
                "p_composite_success_rolewise"
            ),
            war_target=(
                "p_war_success_rolewise"
            ),
            actual_success_col=(
                "p_actual_success"
            ),
            composite_score_col=(
                "p_composite_score_rolewise"
            ),
            playtime_col="post_IP_3y",
        )
    )

    hitter_label_cases.to_csv(
        RESULT_DIR
        / "hitter_player_label_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sp_label_cases.to_csv(
        RESULT_DIR
        / "sp_player_label_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rp_label_cases.to_csv(
        RESULT_DIR
        / "rp_player_label_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 모델 학습 및 actual success 비교
    # -----------------------------------------------------

    metric_frames: list[pd.DataFrame] = []

    player_prediction_frames: list[
        pd.DataFrame
    ] = []

    # 타자
    hitter_metrics, hitter_predictions = (
        run_group_comparison(
            df=hitter,
            features=HITTER_EXPANDED_FEATURES,
            composite_target=(
                "h_composite_success_rolewise"
            ),
            war_target=(
                "h_war_success_rolewise"
            ),
            actual_success_col=(
                "h_actual_success"
            ),
            dataset_name="hitter",
        )
    )

    metric_frames.append(
        hitter_metrics
    )

    player_prediction_frames.append(
        hitter_predictions
    )

    # SP
    sp_metrics, sp_predictions = (
        run_group_comparison(
            df=pitcher_sp,
            features=(
                PITCHER_SP_EXPANDED_FEATURES
            ),
            composite_target=(
                "p_composite_success_rolewise"
            ),
            war_target=(
                "p_war_success_rolewise"
            ),
            actual_success_col=(
                "p_actual_success"
            ),
            dataset_name="pitcher_sp",
        )
    )

    metric_frames.append(
        sp_metrics
    )

    player_prediction_frames.append(
        sp_predictions
    )

    # RP
    rp_metrics, rp_predictions = (
        run_group_comparison(
            df=pitcher_rp,
            features=(
                PITCHER_RP_EXPANDED_FEATURES
            ),
            composite_target=(
                "p_composite_success_rolewise"
            ),
            war_target=(
                "p_war_success_rolewise"
            ),
            actual_success_col=(
                "p_actual_success"
            ),
            dataset_name="pitcher_rp",
        )
    )

    metric_frames.append(
        rp_metrics
    )

    player_prediction_frames.append(
        rp_predictions
    )

    all_metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    all_player_predictions = pd.concat(
        player_prediction_frames,
        ignore_index=True,
    )

    final_summary = create_final_model_summary(
        all_metrics
    )

    difference_table = (
        create_pairwise_difference_table(
            final_summary
        )
    )

    # -----------------------------------------------------
    # 결과 저장
    # -----------------------------------------------------

    all_metrics.to_csv(
        RESULT_DIR
        / "all_detailed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_summary.to_csv(
        RESULT_DIR
        / "final_actual_success_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    difference_table.to_csv(
        RESULT_DIR
        / "composite_minus_war_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_player_predictions.to_csv(
        RESULT_DIR
        / "all_test_player_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Composite가 더 잘 맞힌 사례만 별도 저장
    all_label_cases = pd.concat(
        [
            hitter_label_cases,
            sp_label_cases,
            rp_label_cases,
        ],
        ignore_index=True,
    )

    all_label_cases[
        all_label_cases[
            "composite_better_case"
        ]
    ].to_csv(
        RESULT_DIR
        / "composite_better_player_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_label_cases[
        all_label_cases[
            "war_better_case"
        ]
    ].to_csv(
        RESULT_DIR
        / "war_better_player_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 최종 출력
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("LABEL DEFINITION VS ACTUAL SUCCESS")
    print("=" * 80)

    print(
        label_definition_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    print(
        final_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("COMPOSITE MINUS WAR DIFFERENCE")
    print("=" * 80)

    print(
        difference_table.to_string(
            index=False
        )
    )

    print("\n결과 저장 위치:")

    print(RESULT_DIR)

    print("\n주요 파일:")

    print(
        "1. final_actual_success_comparison.csv"
    )

    print(
        "2. composite_minus_war_differences.csv"
    )

    print(
        "3. label_definition_vs_actual_success.csv"
    )

    print(
        "4. all_test_player_predictions.csv"
    )

    print(
        "5. composite_better_player_cases.csv"
    )

    print(
        "6. war_better_player_cases.csv"
    )

    print(
        "7. role_specific_success_criteria.csv"
    )


if __name__ == "__main__":
    main()