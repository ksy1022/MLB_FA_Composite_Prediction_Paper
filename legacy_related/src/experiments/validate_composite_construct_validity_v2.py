from __future__ import annotations

"""Composite Score의 데이터 내부 구성 타당성 검증.

외부 성공기준을 새 라벨로 사용하지 않고 다음을 검증한다.
1) WAR와 PA/IP가 같은 방향의 공통 성과축을 이루는가
2) 동일가중 Composite가 데이터 기반 PC1과 유사한가
3) 성공 라벨이 가중치 변화와 부트스트랩에 안정적인가
"""

import warnings
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_STATE = 42
N_BOOTSTRAP = 2000
WEIGHT_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "composite_construct_validity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 프로젝트 루트를 import 경로에 추가
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.modeling.final_2_class_model import engineer_pitcher_features
except ImportError as exc:
    engineer_pitcher_features = None
    ENGINEER_IMPORT_ERROR = exc
else:
    ENGINEER_IMPORT_ERROR = None



@dataclass(frozen=True)
class GroupSpec:
    name: str
    playtime_col: str
    success_rate: float


GROUP_SPECS = {
    "hitter": GroupSpec("hitter", "post_PA_3y", 0.40),
    "pitcher_sp": GroupSpec("pitcher_sp", "post_IP_3y", 0.20),
    "pitcher_rp": GroupSpec("pitcher_rp", "post_IP_3y", 0.20),
}


def find_csv(kind: str) -> Path:
    patterns = {
        "hitter": [
            "**/final_hitter_fa_dataset_2011_2024_FAonly_clean_posgroup.csv",
            "**/*hitter*FAonly*clean*.csv",
            "**/*hitter*final*.csv",
        ],
        "pitcher": [
            "**/final_pitcher_fa_dataset_2011_2024_FAonly_clean*.csv",
            "**/*pitcher*FAonly*clean*.csv",
            "**/*pitcher*final*.csv",
        ],
    }
    candidates = []
    for pattern in patterns[kind]:
        candidates.extend(ROOT.glob(pattern))
    candidates = [
        p for p in set(candidates)
        if "outputs" not in p.parts
        and "composite_construct_validity" not in p.parts
    ]
    if not candidates:
        raise FileNotFoundError(
            f"{kind} 최종 데이터 CSV를 찾지 못했습니다. find_csv()에 실제 파일명을 추가하세요."
        )
    return sorted(
        candidates,
        key=lambda p: ("final_" not in p.name.lower(), -p.stat().st_mtime),
    )[0]


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - x.mean()) / sd


def safe_corr(x, y, method="pearson") -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan
    if method == "pearson":
        return float(pearsonr(x[mask], y[mask]).statistic)
    return float(spearmanr(x[mask], y[mask]).statistic)


def orient_pca(pca: PCA, scores: np.ndarray):
    loadings = pca.components_[0].copy()
    if loadings.sum() < 0:
        loadings *= -1
        scores = -scores
    return loadings, scores


def top_label(score: pd.Series, success_rate: float) -> pd.Series:
    n = len(score)
    k = max(1, int(round(n * success_rate)))
    rank = score.rank(method="first", ascending=False)
    return (rank <= k).astype(int)


def jaccard_binary(a: pd.Series, b: pd.Series) -> float:
    aa = a.astype(bool).to_numpy()
    bb = b.astype(bool).to_numpy()
    union = np.logical_or(aa, bb).sum()
    return np.nan if union == 0 else float(np.logical_and(aa, bb).sum() / union)


def cohen_kappa_binary(a: pd.Series, b: pd.Series) -> float:
    aa = a.astype(int).to_numpy()
    bb = b.astype(int).to_numpy()
    po = np.mean(aa == bb)
    pa = np.mean(aa == 1)
    pb = np.mean(bb == 1)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return np.nan if pe == 1 else float((po - pe) / (1 - pe))


def prepare_groups():
    hitter_path = find_csv("hitter")
    pitcher_path = find_csv("pitcher")
    print(f"[HITTER DATA] {hitter_path}")
    print(f"[PITCHER DATA] {pitcher_path}")

    hitter = pd.read_csv(hitter_path)
    pitcher = pd.read_csv(pitcher_path)

    # Role/is_SP/is_RP는 processed CSV에 직접 저장되지 않고,
    # 기존 모델의 feature engineering 단계에서 생성될 수 있다.
    if engineer_pitcher_features is None:
        raise ImportError(
            "src.modeling.final_2_class_model에서 "
            "engineer_pitcher_features를 불러오지 못했습니다. "
            f"원인: {ENGINEER_IMPORT_ERROR}"
        )

    pitcher = engineer_pitcher_features(pitcher.copy())
    print("[PITCHER FEATURE ENGINEERING] 기존 final_2_class_model 함수 적용")
    print(
        "[PITCHER ROLE CANDIDATES]",
        [
            c for c in pitcher.columns
            if any(token in c.lower() for token in [
                "role", "is_sp", "is_rp", "gs_ratio", "starter", "reliever"
            ])
        ],
    )

    for col in ["post_WAR_3y_sum", "post_PA_3y"]:
        if col not in hitter.columns:
            raise KeyError(f"Hitter 데이터 누락 컬럼: {col}")
    for col in ["post_WAR_3y_sum", "post_IP_3y"]:
        if col not in pitcher.columns:
            raise KeyError(f"Pitcher 데이터 누락 컬럼: {col}")

    if "Role" in pitcher.columns:
        pitcher["Role"] = pitcher["Role"].astype(str).str.upper().str.strip()
    elif {"is_SP", "is_RP"}.issubset(pitcher.columns):
        pitcher["Role"] = np.where(
            pd.to_numeric(pitcher["is_SP"], errors="coerce").fillna(0) >= 1,
            "SP",
            "RP",
        )
    else:
        role_candidates = [
            c for c in pitcher.columns
            if any(token in c.lower() for token in ["role", "is_sp", "is_rp", "gs_ratio", "starter", "reliever"])
        ]
        raise KeyError(
            "기존 feature engineering을 적용한 뒤에도 투수 역할 컬럼을 찾지 못했습니다. "
            f"역할 관련 후보 컬럼: {role_candidates}"
        )

    groups = {
        "hitter": hitter.copy(),
        "pitcher_sp": pitcher[pitcher["Role"].eq("SP")].copy(),
        "pitcher_rp": pitcher[pitcher["Role"].eq("RP")].copy(),
    }

    for name, df in groups.items():
        spec = GROUP_SPECS[name]
        cols = ["post_WAR_3y_sum", spec.playtime_col]
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=cols).reset_index(drop=True)
        groups[name] = df
        print(f"[{name}] n={len(df)}")
    return groups


def analyze_group(df: pd.DataFrame, spec: GroupSpec, rng):
    work = df.copy()
    work["z_war"] = zscore(work["post_WAR_3y_sum"])
    work["z_playtime"] = zscore(work[spec.playtime_col])
    work["composite_equal"] = work["z_war"] + work["z_playtime"]

    X = work[["z_war", "z_playtime"]].to_numpy()
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pc1 = pca.fit_transform(X)[:, 0]
    loadings, pc1 = orient_pca(pca, pc1)
    work["pc1_score"] = pc1
    work["equal_success"] = top_label(work["composite_equal"], spec.success_rate)
    work["pc1_success"] = top_label(work["pc1_score"], spec.success_rate)

    summary = pd.DataFrame([{
        "dataset": spec.name,
        "n_players": len(work),
        "component_pearson_r": safe_corr(work["z_war"], work["z_playtime"]),
        "component_spearman_rho": safe_corr(work["z_war"], work["z_playtime"], "spearman"),
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pc1_war_loading": float(loadings[0]),
        "pc1_playtime_loading": float(loadings[1]),
        "pc1_loading_absolute_difference": float(abs(abs(loadings[0]) - abs(loadings[1]))),
        "equal_pc1_pearson_r": safe_corr(work["composite_equal"], work["pc1_score"]),
        "equal_pc1_spearman_rho": safe_corr(work["composite_equal"], work["pc1_score"], "spearman"),
        "success_rate_used": spec.success_rate,
        "equal_pc1_label_agreement": float(np.mean(work["equal_success"] == work["pc1_success"])),
        "equal_pc1_jaccard": jaccard_binary(work["equal_success"], work["pc1_success"]),
        "equal_pc1_cohen_kappa": cohen_kappa_binary(work["equal_success"], work["pc1_success"]),
    }])

    sensitivity_rows = []
    for war_weight in WEIGHT_GRID:
        score = war_weight * work["z_war"] + (1 - war_weight) * work["z_playtime"]
        label = top_label(score, spec.success_rate)
        sensitivity_rows.append({
            "dataset": spec.name,
            "war_weight": float(war_weight),
            "playtime_weight": float(1 - war_weight),
            "spearman_with_equal_score": safe_corr(score, work["composite_equal"], "spearman"),
            "label_agreement_with_equal": float(np.mean(label == work["equal_success"])),
            "jaccard_with_equal_success": jaccard_binary(label, work["equal_success"]),
            "cohen_kappa_with_equal": cohen_kappa_binary(label, work["equal_success"]),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)

    boot_rows = []
    n = len(work)
    for iteration in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        sample = work.iloc[idx].copy()
        zw = zscore(sample["post_WAR_3y_sum"])
        zp = zscore(sample[spec.playtime_col])
        Xb = np.column_stack([zw, zp])
        if np.std(Xb[:, 0]) == 0 or np.std(Xb[:, 1]) == 0:
            continue
        pca_b = PCA(n_components=2, random_state=RANDOM_STATE)
        pc_b = pca_b.fit_transform(Xb)[:, 0]
        load_b, pc_b = orient_pca(pca_b, pc_b)
        equal_b = zw + zp
        equal_lab = top_label(pd.Series(equal_b), spec.success_rate)
        pc_lab = top_label(pd.Series(pc_b), spec.success_rate)
        boot_rows.append({
            "dataset": spec.name,
            "iteration": iteration,
            "component_pearson_r": safe_corr(Xb[:, 0], Xb[:, 1]),
            "pc1_explained_variance_ratio": float(pca_b.explained_variance_ratio_[0]),
            "pc1_war_loading": float(load_b[0]),
            "pc1_playtime_loading": float(load_b[1]),
            "pc1_loading_absolute_difference": float(abs(abs(load_b[0]) - abs(load_b[1]))),
            "equal_pc1_spearman_rho": safe_corr(equal_b, pc_b, "spearman"),
            "equal_pc1_label_agreement": float(np.mean(equal_lab == pc_lab)),
            "equal_pc1_jaccard": jaccard_binary(equal_lab, pc_lab),
        })
    bootstrap = pd.DataFrame(boot_rows)

    year_rows = []
    if "fa_year" in work.columns:
        for year, group in work.groupby("fa_year"):
            if len(group) < 10:
                continue
            eq = top_label(group["composite_equal"], spec.success_rate)
            pc = top_label(group["pc1_score"], spec.success_rate)
            year_rows.append({
                "dataset": spec.name,
                "fa_year": year,
                "n_players": len(group),
                "equal_pc1_spearman_rho": safe_corr(group["composite_equal"], group["pc1_score"], "spearman"),
                "label_agreement": float(np.mean(eq == pc)),
                "jaccard": jaccard_binary(eq, pc),
            })
    yearly = pd.DataFrame(year_rows)

    keep = [c for c in [
        "player_key", "player", "fa_year", "Role", "post_WAR_3y_sum", spec.playtime_col,
        "z_war", "z_playtime", "composite_equal", "pc1_score", "equal_success", "pc1_success"
    ] if c in work.columns]
    work[keep].to_csv(OUTPUT_DIR / f"{spec.name}_player_scores.csv", index=False, encoding="utf-8-sig")
    return summary, sensitivity, bootstrap, yearly


def bootstrap_ci_table(bootstrap):
    metrics = [
        "component_pearson_r", "pc1_explained_variance_ratio", "pc1_war_loading",
        "pc1_playtime_loading", "pc1_loading_absolute_difference",
        "equal_pc1_spearman_rho", "equal_pc1_label_agreement", "equal_pc1_jaccard"
    ]
    rows = []
    for dataset, group in bootstrap.groupby("dataset"):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "bootstrap_mean": float(values.mean()),
                "ci_lower_2.5%": float(values.quantile(0.025)),
                "ci_upper_97.5%": float(values.quantile(0.975)),
                "n_valid": int(len(values)),
            })
    return pd.DataFrame(rows)


def add_diagnostic_flags(summary):
    out = summary.copy()
    out["components_same_direction"] = out["component_pearson_r"] > 0
    out["components_not_redundant"] = out["component_pearson_r"].abs() < 0.90
    out["pc1_majority_variance"] = out["pc1_explained_variance_ratio"] >= 0.60
    out["equal_weights_close_to_pc1"] = out["pc1_loading_absolute_difference"] <= 0.15
    out["equal_score_tracks_pc1"] = out["equal_pc1_spearman_rho"] >= 0.95
    out["labels_highly_stable"] = out["equal_pc1_jaccard"] >= 0.80
    return out


def main():
    print("=" * 88)
    print("COMPOSITE SCORE CONSTRUCT VALIDITY")
    print("=" * 88)
    groups = prepare_groups()
    rng = np.random.default_rng(RANDOM_STATE)

    summaries, sensitivities, bootstraps, yearlies = [], [], [], []
    for name, df in groups.items():
        print("\n" + "=" * 88)
        print(name.upper())
        print("=" * 88)
        result = analyze_group(df, GROUP_SPECS[name], rng)
        summary, sensitivity, bootstrap, yearly = result
        summaries.append(summary)
        sensitivities.append(sensitivity)
        bootstraps.append(bootstrap)
        if not yearly.empty:
            yearlies.append(yearly)
        print(summary.to_string(index=False))

    all_summary = pd.concat(summaries, ignore_index=True)
    all_sensitivity = pd.concat(sensitivities, ignore_index=True)
    all_bootstrap = pd.concat(bootstraps, ignore_index=True)
    all_yearly = pd.concat(yearlies, ignore_index=True) if yearlies else pd.DataFrame()
    ci = bootstrap_ci_table(all_bootstrap)
    diagnostic = add_diagnostic_flags(all_summary)

    all_summary.to_csv(OUTPUT_DIR / "construct_validity_summary.csv", index=False, encoding="utf-8-sig")
    all_sensitivity.to_csv(OUTPUT_DIR / "weight_sensitivity.csv", index=False, encoding="utf-8-sig")
    all_bootstrap.to_csv(OUTPUT_DIR / "bootstrap_raw.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(OUTPUT_DIR / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    diagnostic.to_csv(OUTPUT_DIR / "construct_validity_diagnostic.csv", index=False, encoding="utf-8-sig")
    if not all_yearly.empty:
        all_yearly.to_csv(OUTPUT_DIR / "yearly_stability.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    print(diagnostic.to_string(index=False))
    print("\n결과 저장 위치:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
