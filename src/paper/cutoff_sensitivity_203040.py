"""
Cutoff Sensitivity Analysis
===========================

목적
----
- 기존 2-class 고성과 기준의 민감도 분석
- Hitter / SP / RP에 대해 Top 20%, 30%, 40% 비교
- Composite와 WAR-only 결과 비교
- 기존 논문 결과의 재현 여부 확인

Primary analysis
----------------
- paper_model_spec.json에 지정된 모델 고정
- 검증 세트에서 probability threshold 선택

Secondary analysis
------------------
- 각 cutoff에서 validation 성능을 기준으로 모델 선택

주의
----
- Test 성능을 이용하여 cutoff를 선택하지 않음
- 기존 결과 파일을 덮어쓰지 않음
"""

from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline


# ============================================================
# 1. 경로 및 기본 설정
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from modeling import run_temporal_final_models as base


OUTPUT_DIR = ROOT / "outputs" / "cutoff_sensitivity_203040"
MODEL_SPEC_PATH = ROOT / "config" / "paper_model_spec.json"

TOP_RATIOS = [0.20, 0.30, 0.40]
OUTCOMES = ["composite", "war"]

BOOTSTRAP_REPETITIONS = 10_000
RANDOM_SEED = 42


# ============================================================
# 2. Bootstrap
# ============================================================

def cluster_bootstrap_accuracy_difference(
    paired_data: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int = BOOTSTRAP_REPETITIONS,
) -> dict:
    """
    Composite와 WAR-only의 Accuracy 차이에 대해
    player 단위 cluster bootstrap을 수행한다.

    동일 선수의 여러 계약은 하나의 cluster로 함께 재표집한다.
    """

    grouped = (
        paired_data
        .groupby("player_key")["difference"]
        .agg(["sum", "count"])
    )

    bootstrap_index = rng.integers(
        low=0,
        high=len(grouped),
        size=(n_bootstrap, len(grouped)),
    )

    difference_sum = grouped["sum"].to_numpy()
    observation_count = grouped["count"].to_numpy()

    draws = (
        difference_sum[bootstrap_index].sum(axis=1)
        / observation_count[bootstrap_index].sum(axis=1)
    )

    return {
        "delta_accuracy": paired_data["difference"].mean(),
        "bootstrap_se": draws.std(ddof=1),
        "ci_low": np.quantile(draws, 0.025),
        "ci_high": np.quantile(draws, 0.975),
        "n_players": len(grouped),
    }


# ============================================================
# 3. 역할별 데이터 로드
# ============================================================

def load_role_data(role: str) -> pd.DataFrame:
    """최종 V2 cohort를 불러오고 시간순 split을 부여한다."""

    data_path = base.DATA / f"{role}_cohort_v2.csv"

    df = pd.read_csv(data_path)
    df["split"] = base.temporal_split(df)

    df = df[df["split"].notna()].copy()

    if not df["fa_year"].between(2012, 2021).all():
        raise ValueError(
            f"{role}: 분석기간 밖의 fa_year가 포함되어 있습니다."
        )

    return df


def get_player_overlap(
    df: pd.DataFrame,
    role: str,
) -> list[dict]:
    """Train과 Valid/Test 사이의 동일 선수 중복 수를 확인한다."""

    results = []

    for split_a, split_b in [
        ("train", "valid"),
        ("train", "test"),
    ]:
        players_a = set(
            df.loc[df["split"] == split_a, "player_key"]
        )
        players_b = set(
            df.loc[df["split"] == split_b, "player_key"]
        )

        results.append(
            {
                "role": role,
                "split_a": split_a,
                "split_b": split_b,
                "n_shared_players": len(players_a & players_b),
            }
        )

    return results


# ============================================================
# 4. 라벨 및 분포 정보 기록
# ============================================================

def collect_class_information(
    df: pd.DataFrame,
    role: str,
    top_ratio: float,
    outcome: str,
    target: str,
    cfg: dict,
) -> tuple[list[dict], list[dict]]:
    """
    각 split의 High 표본 수와
    High / Other 집단의 WAR·PA/IP 특성을 기록한다.
    """

    counts = []
    descriptions = []

    for split_name, split_df in df.groupby("split"):

        counts.append(
            {
                "role": role,
                "top": top_ratio,
                "outcome": outcome,
                "split": split_name,
                "n": len(split_df),
                "high": int(split_df[target].sum()),
                "high_rate": split_df[target].mean(),
                "n_players": split_df["player_key"].nunique(),
            }
        )

        for high_value, group in split_df.groupby(target):

            descriptions.append(
                {
                    "role": role,
                    "top": top_ratio,
                    "outcome": outcome,
                    "split": split_name,
                    "high": int(high_value),
                    "n": len(group),
                    "war_median": group[cfg["war"]].median(),
                    "war_q25": group[cfg["war"]].quantile(0.25),
                    "war_q75": group[cfg["war"]].quantile(0.75),
                    "playtime_median": group[cfg["playtime"]].median(),
                    "war_nonpositive_rate": (
                        group[cfg["war"]].le(0).mean()
                    ),
                }
            )

    return counts, descriptions


# ============================================================
# 5. 모델 학습 및 평가
# ============================================================

def run_models_for_outcome(
    df: pd.DataFrame,
    role: str,
    top_ratio: float,
    outcome: str,
    cfg: dict,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """
    하나의 역할 × cutoff × outcome 조합에 대해
    LR / RF / XGB를 학습하고 validation/test 성능을 계산한다.
    """

    target = f"{outcome}_2class"

    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"]
    test = df[df["split"] == "test"]

    majority_class = int(train[target].mode().iloc[0])

    metrics = []
    predictions = []

    models = base.models(
        "2class",
        train[target],
    )

    for model_name, estimator in models.items():

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    base.make_preprocessor(
                        train[cfg["features"]]
                    ),
                ),
                ("model", estimator),
            ]
        )

        pipeline.fit(
            train[cfg["features"]],
            train[target],
        )

        # ----------------------------------------------------
        # Validation에서 probability threshold 결정
        # ----------------------------------------------------

        valid_probability = pipeline.predict_proba(
            valid[cfg["features"]]
        )[:, 1]

        threshold = base.select_threshold(
            valid[target],
            valid_probability,
        )

        # ----------------------------------------------------
        # Validation / Test 평가
        # ----------------------------------------------------

        evaluation_sets = {
            "valid": (valid, valid_probability),
            "test": (
                test,
                pipeline.predict_proba(
                    test[cfg["features"]]
                )[:, 1],
            ),
        }

        for split_name, (
            split_df,
            probability,
        ) in evaluation_sets.items():

            values, prediction = base.binary_metrics(
                split_df[target],
                probability,
                threshold,
            )

            prevalence = split_df[target].mean()

            majority_accuracy = (
                split_df[target] == majority_class
            ).mean()

            ap_lift = (
                values["pr_auc"] / prevalence
                if prevalence > 0
                else np.nan
            )

            metrics.append(
                {
                    "role": role,
                    "top": top_ratio,
                    "outcome": outcome,
                    "model": model_name,
                    "split": split_name,
                    "n": len(split_df),
                    "high": int(split_df[target].sum()),
                    "predicted_high": int(prediction.sum()),
                    "threshold": threshold,
                    "train_majority_accuracy":
                        majority_accuracy,
                    "accuracy_gain":
                        values["accuracy"]
                        - majority_accuracy,
                    "ap_baseline": prevalence,
                    "ap_lift": ap_lift,
                    **values,
                }
            )

            prediction_df = (
                split_df[
                    ["player_key", "fa_year"]
                ]
                .copy()
            )

            prediction_df = prediction_df.assign(
                role=role,
                top=top_ratio,
                outcome=outcome,
                model=model_name,
                split=split_name,
                y_true=split_df[target].to_numpy(),
                y_pred=prediction,
                probability=probability,
            )

            predictions.append(prediction_df)

    return metrics, predictions


# ============================================================
# 6. Paper-fixed 모델 추출
# ============================================================

def select_paper_fixed_models(
    metrics: pd.DataFrame,
    model_spec: dict,
) -> pd.DataFrame:
    """
    논문에서 지정한 역할별 모델만 추출한다.
    """

    mask = []

    for row in metrics.itertuples():
        expected_model = model_spec[row.role]["2class"][
            row.outcome
        ]
        mask.append(row.model == expected_model)

    return metrics.loc[mask].copy()


# ============================================================
# 7. Validation 기반 모델 선택
# ============================================================

def select_validation_models(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    각 role × cutoff × outcome에서
    validation Accuracy가 가장 높은 모델을 선택한다.

    Accuracy 동률이면 Balanced Accuracy를 사용한다.
    """

    validation = metrics[
        metrics["split"] == "valid"
    ].copy()

    selected_models = (
        validation
        .sort_values(
            [
                "role",
                "top",
                "outcome",
                "accuracy",
                "balanced_accuracy",
            ],
            ascending=[
                True,
                True,
                True,
                False,
                False,
            ],
            kind="stable",
        )
        .groupby(
            ["role", "top", "outcome"],
            as_index=False,
        )
        .head(1)[
            ["role", "top", "outcome", "model"]
        ]
    )

    return selected_models.merge(
        metrics,
        on=[
            "role",
            "top",
            "outcome",
            "model",
        ],
        how="left",
    )


# ============================================================
# 8. Composite vs WAR Bootstrap
# ============================================================

def calculate_bootstrap_differences(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    policy_name: str,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Composite와 WAR-only의 test Accuracy 차이를
    역할 및 cutoff별로 계산한다.
    """

    results = []

    test_metrics = metrics[
        metrics["split"] == "test"
    ]

    for (
        role,
        top_ratio,
    ), pair in test_metrics.groupby(
        ["role", "top"]
    ):

        role_predictions = predictions[
            (predictions["role"] == role)
            & (predictions["top"] == top_ratio)
            & (predictions["split"] == "test")
        ]

        outcome_results = []

        for outcome in OUTCOMES:

            model_name = pair.loc[
                pair["outcome"] == outcome,
                "model",
            ].iloc[0]

            arm = role_predictions[
                (role_predictions["outcome"] == outcome)
                & (
                    role_predictions["model"]
                    == model_name
                )
            ].copy()

            arm["correct"] = (
                arm["y_true"] == arm["y_pred"]
            ).astype(int)

            outcome_results.append(
                arm[
                    [
                        "player_key",
                        "fa_year",
                        "correct",
                    ]
                ]
            )

        paired = outcome_results[0].merge(
            outcome_results[1],
            on=["player_key", "fa_year"],
            validate="one_to_one",
            suffixes=("_composite", "_war"),
        )

        paired["difference"] = (
            paired["correct_composite"]
            - paired["correct_war"]
        )

        bootstrap_result = (
            cluster_bootstrap_accuracy_difference(
                paired,
                rng,
            )
        )

        results.append(
            {
                "policy": policy_name,
                "role": role,
                "top": top_ratio,
                **bootstrap_result,
            }
        )

    return results


# ============================================================
# 9. 기존 논문 결과 재현 확인
# ============================================================

def verify_original_predictions(
    predictions: pd.DataFrame,
    model_spec: dict,
) -> pd.DataFrame:
    """
    기존 논문 기준
    - Hitter: Top 40%
    - SP/RP: Top 20%

    에서 기존 저장 결과와 새 분석 결과가 동일한지 확인한다.
    """

    original_path = (
        ROOT
        / "outputs"
        / "temporal_accuracy_final"
        / "temporal_predictions.csv"
    )

    original = pd.read_csv(original_path)

    checks = []

    for role, cfg in base.ROLE_CONFIG.items():

        for outcome in OUTCOMES:

            model_name = model_spec[role]["2class"][
                outcome
            ]

            new = predictions[
                (predictions["role"] == role)
                & (
                    predictions["top"]
                    == cfg["top"]
                )
                & (
                    predictions["outcome"]
                    == outcome
                )
                & (
                    predictions["model"]
                    == model_name
                )
                & (
                    predictions["split"]
                    == "test"
                )
            ]

            old = original[
                (original["role"] == role)
                & (
                    original["outcome"]
                    == outcome
                )
                & (
                    original["model"]
                    == model_name
                )
                & (
                    original["task"]
                    == "2class"
                )
                & (
                    original["split"]
                    == "test"
                )
            ]

            merged = new.merge(
                old,
                on=["player_key", "fa_year"],
                suffixes=("_new", "_old"),
                validate="one_to_one",
            )

            probability_columns = [
                column
                for column in merged.columns
                if column.startswith("probability")
            ]

            max_probability_difference = np.nan

            if len(probability_columns) >= 2:
                max_probability_difference = (
                    merged[
                        probability_columns[0]
                    ]
                    - merged[
                        probability_columns[1]
                    ]
                ).abs().max()

            checks.append(
                {
                    "role": role,
                    "outcome": outcome,
                    "model": model_name,
                    "n": len(merged),
                    "labels_equal": bool(
                        (
                            merged["y_true_new"]
                            == merged["y_true_old"]
                        ).all()
                    ),
                    "predictions_equal": bool(
                        (
                            merged["y_pred_new"]
                            == merged["y_pred_old"]
                        ).all()
                    ),
                    "max_probability_difference":
                        float(
                            max_probability_difference
                        ),
                }
            )

    return pd.DataFrame(checks)


# ============================================================
# 10. 결과 저장
# ============================================================

def save_results(
    output_files: dict[str, pd.DataFrame],
) -> None:
    """분석 결과 CSV를 저장한다."""

    for name, dataframe in output_files.items():

        path = OUTPUT_DIR / f"{name}.csv"

        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )


def save_metadata(
    model_spec: dict,
) -> None:
    """실행환경과 데이터 hash 정보를 저장한다."""

    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": (
            __import__("xgboost").__version__
            if base.XGBClassifier
            else None
        ),
        "bootstrap": BOOTSTRAP_REPETITIONS,
        "seed": RANDOM_SEED,
        "cutoffs": TOP_RATIOS,
        "data_sha256": {
            file.name: hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
            for file in base.DATA.glob(
                "*_cohort_v2.csv"
            )
        },
        "score": (
            "zWAR + zPA/IP "
            "(divide cuts by two for "
            "0.5/0.5 notation)"
        ),
        "primary_models": model_spec,
        "limitation": (
            "Post-hoc cutoff sensitivity analysis "
            "using previously analyzed test data. "
            "The analysis does not establish an "
            "optimal cutoff or external validity. "
            "Bootstrap intervals are conditional "
            "on fitted models and do not include "
            "model refitting."
        ),
    }

    metadata_path = (
        OUTPUT_DIR / "run_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# 11. Main
# ============================================================

def run() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_spec = json.loads(
        MODEL_SPEC_PATH.read_text(
            encoding="utf-8"
        )
    )

    metrics = []
    predictions = []
    counts = []
    descriptions = []
    parameters = []
    labels = []
    overlaps = []

    # --------------------------------------------------------
    # 역할별 분석
    # --------------------------------------------------------

    for role, original_config in (
        base.ROLE_CONFIG.items()
    ):

        raw = load_role_data(role)

        overlaps.extend(
            get_player_overlap(
                raw,
                role,
            )
        )

        for top_ratio in TOP_RATIOS:

            config = {
                **original_config,
                "top": top_ratio,
            }

            df, outcome_parameters = (
                base.fit_outcomes(
                    raw,
                    config,
                )
            )

            parameters.append(
                {
                    "role": role,
                    "top": top_ratio,
                    **outcome_parameters,
                }
            )

            # ------------------------------------------------
            # Composite / WAR 각각 분석
            # ------------------------------------------------

            for outcome in OUTCOMES:

                target = f"{outcome}_2class"

                current_counts, current_desc = (
                    collect_class_information(
                        df=df,
                        role=role,
                        top_ratio=top_ratio,
                        outcome=outcome,
                        target=target,
                        cfg=config,
                    )
                )

                counts.extend(current_counts)
                descriptions.extend(current_desc)

                label_df = (
                    df[
                        [
                            "player_key",
                            "fa_year",
                            "split",
                            target,
                        ]
                    ]
                    .rename(
                        columns={
                            target: "high"
                        }
                    )
                    .assign(
                        role=role,
                        top=top_ratio,
                        outcome=outcome,
                    )
                )

                labels.append(label_df)

                current_metrics, current_predictions = (
                    run_models_for_outcome(
                        df=df,
                        role=role,
                        top_ratio=top_ratio,
                        outcome=outcome,
                        cfg=config,
                    )
                )

                metrics.extend(
                    current_metrics
                )
                predictions.extend(
                    current_predictions
                )

            print(
                f"Finished {role} "
                f"Top {top_ratio:.0%}",
                flush=True,
            )

    # --------------------------------------------------------
    # DataFrame 변환
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(metrics)

    predictions_df = pd.concat(
        predictions,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Primary: 논문 지정 모델
    # --------------------------------------------------------

    paper_fixed_df = (
        select_paper_fixed_models(
            metrics_df,
            model_spec,
        )
    )

    # --------------------------------------------------------
    # Secondary: validation 기반 모델 선택
    # --------------------------------------------------------

    validation_selected_df = (
        select_validation_models(
            metrics_df
        )
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_results = []

    bootstrap_results.extend(
        calculate_bootstrap_differences(
            metrics=paper_fixed_df,
            predictions=predictions_df,
            policy_name="paper_fixed",
            rng=rng,
        )
    )

    bootstrap_results.extend(
        calculate_bootstrap_differences(
            metrics=validation_selected_df,
            predictions=predictions_df,
            policy_name="validation_selected",
            rng=rng,
        )
    )

    bootstrap_df = pd.DataFrame(
        bootstrap_results
    )

    # --------------------------------------------------------
    # 기존 결과 재현 확인
    # --------------------------------------------------------

    reproduction_checks = (
        verify_original_predictions(
            predictions_df,
            model_spec,
        )
    )

    # 기존 결과와 달라졌다면 즉시 중단
    if not (
        reproduction_checks[
            "labels_equal"
        ].all()
        and reproduction_checks[
            "predictions_equal"
        ].all()
    ):
        raise RuntimeError(
            "기존 논문 결과와 재현 결과가 "
            "일치하지 않습니다."
        )

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    output_files = {
        "all_metrics":
            metrics_df,

        "paper_fixed_metrics":
            paper_fixed_df,

        "validation_selected_metrics":
            validation_selected_df,

        "predictions":
            predictions_df,

        "class_counts":
            pd.DataFrame(counts),

        "outcome_profiles":
            pd.DataFrame(descriptions),

        "label_parameters":
            pd.DataFrame(parameters),

        "labels":
            pd.concat(
                labels,
                ignore_index=True,
            ),

        "cluster_bootstrap_deltas":
            bootstrap_df,

        "original_reproduction_checks":
            reproduction_checks,

        "player_overlap":
            pd.DataFrame(overlaps),
    }

    save_results(output_files)

    save_metadata(model_spec)

    # --------------------------------------------------------
    # 주요 결과 출력
    # --------------------------------------------------------

    columns_to_print = [
        "role",
        "top",
        "outcome",
        "model",
        "accuracy",
        "balanced_accuracy",
        "recall",
        "f1",
        "pr_auc",
    ]

    print("\n=== PAPER FIXED TEST RESULTS ===")

    print(
        paper_fixed_df[
            paper_fixed_df["split"]
            == "test"
        ][columns_to_print]
        .sort_values(
            ["role", "top", "outcome"]
        )
        .to_string(index=False)
    )

    print(
        "\nCutoff sensitivity analysis "
        "completed."
    )

    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    run()