# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
DATA_DIR = ROOT / "data" / "processed_v2"
OUT_DIR = ROOT / "outputs" / "aav_external_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOOT_N = 10000
SEED = 42
TRAIN_START = 2012
TRAIN_END = 2017

ROLE_CONFIG = {
    "Hitter": {"file": "hitter_cohort_v2.csv", "playtime": "post_PA_3y"},
    "Starting Pitcher": {"file": "sp_cohort_v2.csv", "playtime": "post_IP_3y"},
    "Relief Pitcher": {"file": "rp_cohort_v2.csv", "playtime": "post_IP_3y"},
}

def add_composite(frame: pd.DataFrame, playtime_col: str):
    out = frame.copy()
    for c in ["fa_year", "AAV", "post_WAR_3y_sum", playtime_col]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    train = out["fa_year"].between(TRAIN_START, TRAIN_END)
    params = {}
    for col, short in [("post_WAR_3y_sum", "war"), (playtime_col, "playtime")]:
        med = float(out.loc[train, col].median())
        filled_train = out.loc[train, col].fillna(med)
        mean = float(filled_train.mean())
        sd = float(filled_train.std(ddof=0))
        out[f"z_{short}"] = (out[col].fillna(med) - mean) / sd
        params.update({f"{short}_median": med, f"{short}_mean": mean, f"{short}_sd": sd})
    out["composite_score"] = 0.5*out["z_war"] + 0.5*out["z_playtime"]
    return out, params

def bootstrap_delta_rho(df: pd.DataFrame):
    use = df.dropna(subset=["AAV","post_WAR_3y_sum","composite_score"]).copy()
    use = use[use["AAV"] > 0]
    rng = np.random.default_rng(SEED)
    deltas = []

    cluster_col = "player_key" if "player_key" in use.columns and use["player_key"].notna().any() else None
    if cluster_col:
        clusters = use[cluster_col].dropna().unique()
        grouped = {k: use.loc[use[cluster_col] == k] for k in clusters}
        for _ in range(BOOT_N):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            boot = pd.concat([grouped[k] for k in sampled], ignore_index=True)
            rw = spearmanr(boot["AAV"], boot["post_WAR_3y_sum"]).statistic
            rc = spearmanr(boot["AAV"], boot["composite_score"]).statistic
            if np.isfinite(rw) and np.isfinite(rc):
                deltas.append(rc-rw)
        unit = "player_key"
    else:
        n = len(use)
        for _ in range(BOOT_N):
            boot = use.iloc[rng.integers(0, n, size=n)]
            rw = spearmanr(boot["AAV"], boot["post_WAR_3y_sum"]).statistic
            rc = spearmanr(boot["AAV"], boot["composite_score"]).statistic
            if np.isfinite(rw) and np.isfinite(rc):
                deltas.append(rc-rw)
        unit = "row"

    rw = float(spearmanr(use["AAV"], use["post_WAR_3y_sum"]).statistic)
    rc = float(spearmanr(use["AAV"], use["composite_score"]).statistic)
    arr = np.asarray(deltas)
    return {
        "n": len(use),
        "bootstrap_unit": unit,
        "rho_AAV_WAR": rw,
        "rho_AAV_Composite": rc,
        "delta_rho_Composite_minus_WAR": rc-rw,
        "ci_low_2.5": float(np.quantile(arr, .025)),
        "ci_high_97.5": float(np.quantile(arr, .975)),
        "bootstrap_reps_valid": len(arr),
    }

def year_adjusted_regression(df: pd.DataFrame, metric: str):
    use = df[["AAV","fa_year",metric]].dropna().copy()
    use = use[use["AAV"] > 0]
    use["log_AAV"] = np.log(use["AAV"])
    use["metric_z"] = (use[metric]-use[metric].mean())/use[metric].std(ddof=0)
    base = smf.ols("log_AAV ~ C(fa_year)", data=use).fit(cov_type="HC3")
    model = smf.ols("log_AAV ~ metric_z + C(fa_year)", data=use).fit(cov_type="HC3")
    ci = model.conf_int().loc["metric_z"]
    return {
        "n": int(model.nobs),
        "std_beta_on_log_AAV": float(model.params["metric_z"]),
        "robust_se_HC3": float(model.bse["metric_z"]),
        "p_value": float(model.pvalues["metric_z"]),
        "ci_low_95": float(ci.iloc[0]),
        "ci_high_95": float(ci.iloc[1]),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "year_only_r2": float(base.rsquared),
        "incremental_r2_over_year": float(model.rsquared-base.rsquared),
    }

def main():
    corr_rows, boot_rows, reg_rows, param_rows = [], [], [], []

    for role, cfg in ROLE_CONFIG.items():
        path = DATA_DIR / cfg["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run from the project root.")
        df = pd.read_csv(path)
        df, params = add_composite(df, cfg["playtime"])
        df = df[df["AAV"].notna() & (df["AAV"] > 0)].copy()

        for k,v in params.items():
            param_rows.append({"role": role, "parameter": k, "value": v})

        for label, metric in [("WAR","post_WAR_3y_sum"),("Composite","composite_score")]:
            temp = df[["AAV",metric]].dropna()
            res = spearmanr(temp["AAV"], temp[metric])
            corr_rows.append({
                "role": role, "metric": label, "n": len(temp),
                "spearman_rho_with_AAV": float(res.statistic),
                "p_value": float(res.pvalue)
            })

        b = bootstrap_delta_rho(df)
        b["role"] = role
        boot_rows.append(b)

        for label, metric in [("WAR","post_WAR_3y_sum"),("Composite","composite_score")]:
            r = year_adjusted_regression(df, metric)
            r["role"] = role
            r["metric_label"] = label
            reg_rows.append(r)

    corr_df = pd.DataFrame(corr_rows)
    boot_df = pd.DataFrame(boot_rows)
    reg_df = pd.DataFrame(reg_rows)
    params_df = pd.DataFrame(param_rows)

    corr_df.to_csv(OUT_DIR/"01_spearman_aav_association.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(OUT_DIR/"02_bootstrap_delta_spearman.csv", index=False, encoding="utf-8-sig")
    reg_df.to_csv(OUT_DIR/"03_year_adjusted_log_aav_regression.csv", index=False, encoding="utf-8-sig")
    params_df.to_csv(OUT_DIR/"04_composite_training_parameters.csv", index=False, encoding="utf-8-sig")

    print("=== Spearman ===")
    print(corr_df.to_string(index=False))
    print("\n=== Bootstrap delta rho ===")
    print(boot_df.to_string(index=False))
    print("\n=== Year-adjusted log(AAV) regression ===")
    print(reg_df[["role","metric_label","n","std_beta_on_log_AAV","p_value","ci_low_95","ci_high_95","r2","incremental_r2_over_year"]].to_string(index=False))
    print(f"\nSaved to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
