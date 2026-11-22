from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ============================================================
# 1. 경로 설정
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed_v2"
OUTPUT_DIR = ROOT / "outputs" / "paper"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 역할별 설정
# ============================================================

CONFIG = {
    "hitter": {
        "label": "Hitter",
        "file": "hitter_cohort_v2.csv",
        "util_col": "post_PA_3y",
        "high_rate": 0.40,
    },
    "sp": {
        "label": "Starting Pitcher",
        "file": "sp_cohort_v2.csv",
        "util_col": "post_IP_3y",
        "high_rate": 0.20,
    },
    "rp": {
        "label": "Relief Pitcher",
        "file": "rp_cohort_v2.csv",
        "util_col": "post_IP_3y",
        "high_rate": 0.20,
    },
}


# ============================================================
# 3. 2-class High label
# ============================================================

def make_binary_label(
    score: pd.Series,
    high_rate: float,
) -> pd.Series:
    """
    2-class 기준

    Hitter:
        상위 40% = High

    Starting Pitcher / Relief Pitcher:
        상위 20% = High
    """

    k = max(
        1,
        int(round(len(score) * high_rate)),
    )

    label = (
        score.rank(
            method="first",
            ascending=False,
        )
        <= k
    ).astype(int)

    return label


# ============================================================
# 4. 3-class label
# ============================================================

def make_three_class_label(
    score: pd.Series,
) -> pd.Series:
    """
    3-class 기준

    1/3 분위수 미만 = Low
    1/3 ~ 2/3 = Middle
    2/3 이상 = High
    """

    q1 = score.quantile(1 / 3)
    q2 = score.quantile(2 / 3)

    label = np.select(
        [
            score < q1,
            score < q2,
        ],
        [
            "Low",
            "Middle",
        ],
        default="High",
    )

    return pd.Series(
        label,
        index=score.index,
        dtype="object",
    )


# ============================================================
# 5. 역할별 분석
# ============================================================

def analyze_role(
    role_key: str,
    config: dict,
) -> dict:

    file_path = (
        DATA_DIR
        / config["file"]
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다:\n"
            f"{file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    # --------------------------------------------------------
    # 논문 최종 분석기간
    # --------------------------------------------------------

    if "fa_year" in df.columns:

        df = df[
            df["fa_year"].between(
                2012,
                2021,
            )
        ].copy()

    war_col = "post_WAR_3y_sum"
    util_col = config["util_col"]

    required_cols = [
        war_col,
        util_col,
    ]

    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:

        raise KeyError(
            f"[{config['label']}] "
            f"필요한 컬럼이 없습니다.\n"
            f"Missing: {missing}\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # 필요한 데이터만 사용
    # --------------------------------------------------------

    x = (
        df[
            [
                war_col,
                util_col,
            ]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna()
        .reset_index(drop=True)
    )

    # ========================================================
    # 6. 표준화
    # ========================================================

    z = (
        x - x.mean()
    ) / x.std(
        ddof=0
    )

    z_war = z.iloc[:, 0]
    z_util = z.iloc[:, 1]

    # ========================================================
    # 7. PCA
    # ========================================================

    pca = PCA(
        n_components=2
    )

    pca.fit(z)

    pc1_variance = float(
        pca.explained_variance_ratio_[0]
    )

    loadings = (
        pca.components_[0]
        .astype(float)
        .copy()
    )

    # PCA 방향은 임의이므로
    # 양수 방향으로 통일
    if loadings.sum() < 0:
        loadings *= -1

    war_loading = float(
        loadings[0]
    )

    util_loading = float(
        loadings[1]
    )

    # ========================================================
    # 8. Equal-weight Composite
    # ========================================================

    equal_score = (
        0.5 * z_war
        + 0.5 * z_util
    )

    # baseline labels
    base_binary = make_binary_label(
        equal_score,
        config["high_rate"],
    )

    base_three = make_three_class_label(
        equal_score
    )

    # ========================================================
    # 9. Weight sensitivity
    # ========================================================

    results = {}

    for war_weight in [
        0.4,
        0.6,
    ]:

        util_weight = (
            1.0 - war_weight
        )

        alt_score = (
            war_weight * z_war
            + util_weight * z_util
        )

        # ----------------------------------------------------
        # 2-class
        # ----------------------------------------------------

        alt_binary = make_binary_label(
            alt_score,
            config["high_rate"],
        )

        binary_agreement = float(
            (
                base_binary.to_numpy()
                == alt_binary.to_numpy()
            ).mean()
        )

        # ----------------------------------------------------
        # 3-class
        # ----------------------------------------------------

        alt_three = make_three_class_label(
            alt_score
        )

        three_agreement = float(
            (
                base_three.to_numpy()
                == alt_three.to_numpy()
            ).mean()
        )

        key = (
            f"{war_weight:.1f}:"
            f"{util_weight:.1f}"
        )

        results[
            f"2-class Label Agreement ({key})"
        ] = binary_agreement

        results[
            f"3-class Label Agreement ({key})"
        ] = three_agreement

    # ========================================================
    # 결과 반환
    # ========================================================

    return {
        "Role": config["label"],
        "N": len(x),
        "PC1 Variance Explained": pc1_variance,
        "WAR Loading": war_loading,
        "Utilization Loading": util_loading,
        **results,
    }


# ============================================================
# 10. Main
# ============================================================

def main():

    rows = []

    for role_key, config in CONFIG.items():

        result = analyze_role(
            role_key,
            config,
        )

        rows.append(
            result
        )

    result_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "table8_sensitivity_recomputed.csv"
    )

    result_df.to_csv(
        output_file,
        index=False,
    )

    # --------------------------------------------------------
    # 콘솔 출력
    # --------------------------------------------------------

    print()
    print("=" * 140)

    print(
        "Expanded Table 8:"
        " Construct Validity and Weight Sensitivity"
    )

    print("=" * 140)

    print(
        result_df
        .round(3)
        .to_string(
            index=False
        )
    )

    print("=" * 140)

    print()
    print(
        f"Saved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()