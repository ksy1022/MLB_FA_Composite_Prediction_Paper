"""Finish the GMM structure report from already-computed CSV outputs.

Use this only if check_composite_bimodality.py completed all bootstraps but
crashed while building COMPOSITE_STRUCTURE_REPORT.md with:
KeyError: 'clean_2group_support'

This script does NOT refit any GMM and does NOT rerun bootstrap.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "composite_distribution_structure"


def fmt_num(value, digits=3):
    return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"


def fmt_pct(value):
    return "NA" if pd.isna(value) else f"{float(value):.1%}"


def main():
    combined_path = OUT_DIR / "gmm_combined_decision_summary.csv"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"Missing {combined_path}. "
            "The original analysis must reach the CSV-saving stage first."
        )

    df = pd.read_csv(combined_path)

    required = [
        "role_label",
        "delta_bic_1_minus_2",
        "best_k_bic",
        "two_component_mode_count",
        "ashman_d",
        "boundary",
        "high_rate_at_boundary",
        "p_bic2_better_than_1",
        "p_two_component_density_bimodal",
        "clean_2group_support",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in combined summary: {missing}")

    lines = [
        "# Composite Score distribution-structure check",
        "",
        "Only 2012–2017 training contracts were used to inspect the outcome distribution.",
        "No validation/test prediction metric was used to define the candidate binary boundary.",
        "",
        "## Results",
        "",
        "| Role | ΔBIC 1−2 | Best k (BIC) | Modes in 2-GMM | Ashman D | Boundary | High rate | Bootstrap P(BIC2<1) | Bootstrap P(bimodal) | Clean 2-group support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for _, r in df.iterrows():
        lines.append(
            f"| {r['role_label']} | "
            f"{fmt_num(r['delta_bic_1_minus_2'],1)} | "
            f"{int(r['best_k_bic'])} | "
            f"{int(r['two_component_mode_count'])} | "
            f"{fmt_num(r['ashman_d'],2)} | "
            f"{fmt_num(r['boundary'],3)} | "
            f"{fmt_pct(r['high_rate_at_boundary'])} | "
            f"{fmt_pct(r['p_bic2_better_than_1'])} | "
            f"{fmt_pct(r['p_two_component_density_bimodal'])} | "
            f"{bool(r['clean_2group_support'])} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- `best_k_bic = 3` means a 3-component Gaussian mixture fit has lower BIC than both 1- and 2-component fits.",
        "- `two_component_mode_count = 1` means the fitted 2-component mixture is still visually unimodal; a second Gaussian component may only be modeling skewness or a tail.",
        "- `Ashman D >= 2` is used here only as an empirical separation check.",
        "- `clean_2group_support=False` means the script's conservative criteria do not support claiming a stable natural binary split.",
        "",
    ]

    report_path = OUT_DIR / "COMPOSITE_STRUCTURE_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    cols = required
    print(df[cols].to_string(index=False))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
