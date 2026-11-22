# -*- coding: utf-8 -*-
"""
Characterize the CURRENT paper's role-specific 2-class cutoffs.

Current paper rule
------------------
Hitter            : top 40% of TRAIN Composite Score = High
Starting Pitcher  : top 20%
Relief Pitcher    : top 20%

Purpose
-------
This script does NOT optimize or change the cutoff.
It describes what the current 40/20/20 rule actually means in the
2012-2017 training data by reporting:

- Train N
- Composite skewness
- percentile cutoff
- High / Other counts
- Composite median
- post-contract WAR median
- post-contract PA/IP median
- median differences between High and Other
- IQRs for WAR and PA/IP

Composite is computed exactly as stated in the paper:
    0.5*z(WAR) + 0.5*z(Utilization)

Because the paper's main modeling code uses zWAR + zUtil, the numerical
Composite cutoff there is exactly 2x the value reported here, but rankings
and High/Other membership are identical.

Outputs
-------
outputs/paper/cutoff_characterization/
    cutoff_characterization_table.csv
    cutoff_group_profiles_long.csv
    cutoff_characterization_table.md
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed_v2"
OUT = ROOT / "outputs" / "paper" / "cutoff_characterization"

ROLE_CONFIG = {
    "hitter": {
        "label": "Hitter",
        "file": "hitter_cohort_v2.csv",
        "war": "post_WAR_3y_sum",
        "util": "post_PA_3y",
        "util_label": "PA",
        "top": 0.40,
    },
    "sp": {
        "label": "Starting Pitcher",
        "file": "sp_cohort_v2.csv",
        "war": "post_WAR_3y_sum",
        "util": "post_IP_3y",
        "util_label": "IP",
        "top": 0.20,
    },
    "rp": {
        "label": "Relief Pitcher",
        "file": "rp_cohort_v2.csv",
        "war": "post_WAR_3y_sum",
        "util": "post_IP_3y",
        "util_label": "IP",
        "top": 0.20,
    },
}


def q25(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").quantile(0.25))


def q75(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").quantile(0.75))


def median(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").median())


def fmt(v: float, digits: int = 3) -> str:
    if pd.isna(v):
        return "NA"
    return f"{float(v):.{digits}f}"


def make_train(cfg: dict) -> tuple[pd.DataFrame, float]:
    path = DATA / cfg["file"]
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path)
    df["fa_year"] = pd.to_numeric(df["fa_year"], errors="coerce")
    train = df.loc[df["fa_year"].between(2012, 2017)].copy()

    for col in (cfg["war"], cfg["util"]):
        train[col] = pd.to_numeric(train[col], errors="coerce")

    # Train-only standardization, matching the paper's Figure 1 convention.
    for col, short in ((cfg["war"], "war"), (cfg["util"], "util")):
        med = float(train[col].median())
        filled = train[col].fillna(med)
        mean = float(filled.mean())
        sd = float(filled.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Invalid SD for {cfg['label']} / {col}: {sd}")
        train[f"z_{short}"] = (filled - mean) / sd

    train["composite_score"] = (
        0.5 * train["z_war"] + 0.5 * train["z_util"]
    )

    cutoff = float(train["composite_score"].quantile(1.0 - cfg["top"]))
    train["group"] = np.where(
        train["composite_score"].ge(cutoff), "High", "Other"
    )
    return train, cutoff


def group_profile(train: pd.DataFrame, cfg: dict, group: str) -> dict:
    g = train.loc[train["group"].eq(group)].copy()
    return {
        "role": cfg["label"],
        "group": group,
        "n": int(len(g)),
        "share": float(len(g) / len(train)),
        "composite_median": median(g["composite_score"]),
        "composite_q25": q25(g["composite_score"]),
        "composite_q75": q75(g["composite_score"]),
        "war_median": median(g[cfg["war"]]),
        "war_q25": q25(g[cfg["war"]]),
        "war_q75": q75(g[cfg["war"]]),
        "utilization_metric": cfg["util_label"],
        "utilization_median": median(g[cfg["util"]]),
        "utilization_q25": q25(g[cfg["util"]]),
        "utilization_q75": q75(g[cfg["util"]]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    profile_rows = []

    for role, cfg in ROLE_CONFIG.items():
        train, cutoff = make_train(cfg)
        high = group_profile(train, cfg, "High")
        other = group_profile(train, cfg, "Other")
        profile_rows.extend([high, other])

        rows.append({
            "role": cfg["label"],
            "train_n": int(len(train)),
            "top_ratio": cfg["top"],
            "composite_skewness": float(train["composite_score"].skew()),
            "composite_cutoff": cutoff,
            "high_n": high["n"],
            "high_rate": high["share"],
            "other_n": other["n"],
            "high_composite_median": high["composite_median"],
            "other_composite_median": other["composite_median"],
            "high_war_median": high["war_median"],
            "other_war_median": other["war_median"],
            "war_median_difference": high["war_median"] - other["war_median"],
            "utilization_metric": cfg["util_label"],
            "high_utilization_median": high["utilization_median"],
            "other_utilization_median": other["utilization_median"],
            "utilization_median_difference": (
                high["utilization_median"] - other["utilization_median"]
            ),
            "high_war_q25": high["war_q25"],
            "high_war_q75": high["war_q75"],
            "other_war_q25": other["war_q25"],
            "other_war_q75": other["war_q75"],
            "high_util_q25": high["utilization_q25"],
            "high_util_q75": high["utilization_q75"],
            "other_util_q25": other["utilization_q25"],
            "other_util_q75": other["utilization_q75"],
        })

    summary = pd.DataFrame(rows)
    profiles = pd.DataFrame(profile_rows)

    summary.to_csv(
        OUT / "cutoff_characterization_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    profiles.to_csv(
        OUT / "cutoff_group_profiles_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    md = []
    md.append("# Current 40/20/20 cutoff characterization")
    md.append("")
    md.append(
        "The table characterizes, rather than optimizes, the paper's current "
        "role-specific High definitions using 2012–2017 training data only."
    )
    md.append("")
    md.append(
        "| Role | Train N | Skewness | High rule | Composite cutoff | "
        "High N | High WAR median | Other WAR median | "
        "High PA/IP median | Other PA/IP median |"
    )
    md.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for r in summary.itertuples():
        md.append(
            f"| {r.role} | {r.train_n} | {fmt(r.composite_skewness)} | "
            f"Top {r.top_ratio:.0%} | {fmt(r.composite_cutoff)} | "
            f"{r.high_n} | {fmt(r.high_war_median, 1)} | "
            f"{fmt(r.other_war_median, 1)} | "
            f"{fmt(r.high_utilization_median, 1)} {r.utilization_metric} | "
            f"{fmt(r.other_utilization_median, 1)} {r.utilization_metric} |"
        )

    md.append("")
    md.append("## Suggested interpretation")
    md.append("")
    md.append(
        "These cutoffs should be described as role-specific, distribution-based "
        "relative high-performance definitions. The table shows the empirical "
        "WAR and utilization profiles induced by the current thresholds. It "
        "does not establish that 40% or 20% is a universal or statistically "
        "optimal MLB success threshold."
    )

    (OUT / "cutoff_characterization_table.md").write_text(
        "\n".join(md), encoding="utf-8"
    )

    print("\nCURRENT 40/20/20 CUTOFF CHARACTERIZATION")
    print("=" * 100)
    show = summary[[
        "role", "train_n", "composite_skewness", "top_ratio",
        "composite_cutoff", "high_n", "other_n",
        "high_war_median", "other_war_median",
        "utilization_metric", "high_utilization_median",
        "other_utilization_median",
    ]]
    print(show.to_string(index=False))
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
