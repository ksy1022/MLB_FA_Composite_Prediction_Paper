
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "paper"
TEMP = ROOT / "outputs" / "temporal_accuracy_final"
CONTRIB = ROOT / "outputs" / "final_model_contributions"

ROLES = {
    "hitter": ("Hitter", 534, 328, 86, 120),
    "sp": ("Starting Pitcher", 216, 115, 36, 65),
    "rp": ("Relief Pitcher", 412, 241, 83, 88),
}

# Paper Table 5: 3-class Composite
EXPECTED_3 = {
    "Hitter": (0.483, 0.479, 0.697, 0.648),
    "Starting Pitcher": (0.385, 0.364, 0.333, 0.341),
    "Relief Pitcher": (0.420, 0.398, 0.238, 0.286),
}

# Paper Table 6: 2-class Composite
EXPECTED_2 = {
    "Hitter": ("logistic", .70, .783, .724, .538, .618, .716),
    "Starting Pitcher": ("logistic", .71, .723, .267, .364, .308, .356),
    "Relief Pitcher": ("rf", .64, .898, .000, .000, .000, .176),
}

# Paper Table 7
# Composite Acc, WAR Acc, displayed difference,
# bootstrap mean, SE, CI lower, CI upper
EXPECTED_BOOT = {
    ("Hitter", "2class"):
        (.783, .650, .133, .134, .056, .025, .242),
    ("Hitter", "3class"):
        (.483, .458, .025, .026, .060, -.092, .142),
    ("Starting Pitcher", "2class"):
        (.723, .646, .077, .077, .045, -.015, .169),
    ("Starting Pitcher", "3class"):
        (.385, .369, .016, .014, .077, -.138, .169),
    ("Relief Pitcher", "2class"):
        (.898, .852, .046, .046, .032, -.011, .114),
    ("Relief Pitcher", "3class"):
        (.420, .500, -.080, -.079, .071, -.216, .057),
}

# Paper Table 8: AAV
EXPECTED_AAV = {
    "Hitter": (.395, .587, .192, .143, .244),
    "Starting Pitcher": (.372, .467, .095, .037, .157),
    "Relief Pitcher": (.209, .336, .127, .078, .180),
}

# Paper Table 4
EXPECTED_CUTOFF = {
    "Hitter": (-.028, 3.3, -.2, 1295.0, 327.0),
    "Starting Pitcher": (.802, 8.7, .3, 535.0, 194.5),
    "Relief Pitcher": (.603, 3.6, .0, 206.5, 69.75),
}

# All 12 selected role / outcome / task models
EXPECTED_MODELS = {
    ("hitter", "composite", "2class"): "logistic",
    ("hitter", "composite", "3class"): "rf",
    ("hitter", "war", "2class"): "logistic",
    ("hitter", "war", "3class"): "logistic",
    ("sp", "composite", "2class"): "logistic",
    ("sp", "composite", "3class"): "rf",
    ("sp", "war", "2class"): "logistic",
    ("sp", "war", "3class"): "rf",
    ("rp", "composite", "2class"): "rf",
    ("rp", "composite", "3class"): "rf",
    ("rp", "war", "2class"): "rf",
    ("rp", "war", "3class"): "logistic",
}

results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition), str(detail)))


def eq(name, actual, expected, digits=3):
    try:
        a = float(actual)
        e = float(expected)
        passed = (
            np.isfinite(a)
            and round(a, digits) == round(e, digits)
        )
        check(
            name, passed,
            f"actual={a:.6f}, expected={e:.6f}"
        )
    except (ValueError, TypeError):
        check(name, False, f"Invalid number: {actual}")


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return pd.read_csv(path)


def one_row(df, **conditions):
    selected = df

    for col, value in conditions.items():
        selected = selected[selected[col].eq(value)]

    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {conditions}, got {len(selected)}"
        )

    return selected.iloc[0]


def section(name, fn):
    try:
        fn()
    except Exception as exc:
        check(name, False, f"{type(exc).__name__}: {exc}")


def verify_cohorts():
    for role, (name, n, ntrain, nvalid, ntest) in ROLES.items():
        df = read_csv(
            ROOT / "data" / "processed_v2"
            / f"{role}_cohort_v2.csv"
        )

        eq(f"Cohort {name}", len(df), n, 0)

        year = pd.to_numeric(df["fa_year"])
        eq(f"{name} Train N",
           year.between(2012, 2017).sum(), ntrain, 0)
        eq(f"{name} Valid N",
           year.between(2018, 2019).sum(), nvalid, 0)
        eq(f"{name} Test N",
           year.between(2020, 2021).sum(), ntest, 0)

        check(
            f"{name} unique contract keys",
            not df.duplicated(["player_key", "fa_year"]).any(),
        )

    eq(
        "Total cohort N",
        sum(v[1] for v in ROLES.values()), 1162, 0
    )


def verify_cutoffs():
    df = read_csv(
        OUT / "cutoff_characterization"
        / "cutoff_characterization_table.csv"
    )

    for name, expected in EXPECTED_CUTOFF.items():
        r = one_row(df, role=name)

        columns = [
            "composite_cutoff",
            "high_war_median",
            "other_war_median",
            "high_utilization_median",
            "other_utilization_median",
        ]

        for col, value in zip(columns, expected):
            digits = 2 if col.endswith("utilization_median") else 3
            eq(f"Table 4 {name} {col}", r[col], value, digits)


def verify_model_selection():
    metrics = read_csv(TEMP / "temporal_metrics.csv")
    selected = read_csv(OUT / "paper_selected_models.csv")

    valid = metrics[
        metrics["split"].eq("valid")
        & metrics["model"].isin(["logistic", "rf"])
    ].copy()

    best = (
        valid.sort_values(
            [
                "role", "outcome", "task",
                "accuracy", "balanced_accuracy", "model"
            ],
            ascending=[True, True, True, False, False, True],
        )
        .drop_duplicates(["role", "outcome", "task"])
    )

    eq("Selected model count", len(selected), 12, 0)

    for (role, outcome, task), expected in EXPECTED_MODELS.items():
        r = one_row(
            best, role=role, outcome=outcome, task=task
        )
        s = one_row(
            selected, role=role, outcome=outcome, task=task
        )

        check(
            f"Validation model {role}/{outcome}/{task}",
            r["model"] == expected,
            f"actual={r['model']}, expected={expected}"
        )
        check(
            f"Saved model {role}/{outcome}/{task}",
            s["model"] == r["model"],
        )


def verify_classification():
    t3 = read_csv(OUT / "table4_three_class_composite.csv")
    t2 = read_csv(OUT / "table5_binary_composite.csv")
    pred = read_csv(TEMP / "temporal_predictions.csv")

    for role, (name, _, _, _, ntest) in ROLES.items():
        # 3-class Table 5
        r = one_row(t3, Role=name)
        check(f"3-class model {name}", r["Selected model"] == "rf")

        cols3 = ["Accuracy", "Macro-F1", "High Recall", "High F1"]
        for col, exp in zip(cols3, EXPECTED_3[name]):
            eq(f"Table 5 {name} {col}", r[col], exp)

        # 2-class Table 6
        r2 = one_row(t2, Role=name)
        expected2 = EXPECTED_2[name]

        check(
            f"2-class model {name}",
            r2["Selected model"] == expected2[0],
        )

        cols2 = [
            "Threshold", "Accuracy", "Precision",
            "Recall", "F1", "PR-AUC",
        ]
        for col, exp in zip(cols2, expected2[1:]):
            eq(f"Table 6 {name} {col}", r2[col], exp, 2 if col == "Threshold" else 3)

        # Independently calculate scores from test predictions
        for task, model in [
            ("2class", expected2[0]), ("3class", "rf")
        ]:
            p = pred[
                pred["role"].eq(role)
                & pred["outcome"].eq("composite")
                & pred["task"].eq(task)
                & pred["model"].eq(model)
                & pred["split"].eq("test")
            ]

            eq(f"Prediction N {name}/{task}", len(p), ntest, 0)

            y = p["y_true"].astype(int)
            yhat = p["y_pred"].astype(int)

            expected_acc = (
                expected2[2] if task == "2class"
                else EXPECTED_3[name][0]
            )

            eq(
                f"Recomputed Accuracy {name}/{task}",
                accuracy_score(y, yhat),
                expected_acc,
            )

            if task == "2class":
                eq(
                    f"Recomputed F1 {name}",
                    f1_score(y, yhat, zero_division=0),
                    expected2[5],
                )
                eq(
                    f"Recomputed AP {name}",
                    average_precision_score(
                        y, p["probability_1"]
                    ),
                    expected2[6],
                )
            else:
                eq(
                    f"Recomputed Macro-F1 {name}",
                    f1_score(
                        y, yhat, labels=[0, 1, 2],
                        average="macro", zero_division=0
                    ),
                    EXPECTED_3[name][1],
                )
                eq(
                    f"Recomputed High Recall {name}",
                    recall_score(
                        y, yhat, labels=[2],
                        average=None, zero_division=0
                    )[0],
                    EXPECTED_3[name][2],
                )
                eq(
                    f"Recomputed High F1 {name}",
                    f1_score(
                        y, yhat, labels=[2],
                        average=None, zero_division=0
                    )[0],
                    EXPECTED_3[name][3],
                )


def verify_bootstrap():
    df = read_csv(
        OUT / "table6_composite_vs_war_bootstrap.csv"
    )

    eq("Bootstrap comparison rows", len(df), 6, 0)

    cols = [
        "Composite Accuracy",
        "WAR-only Accuracy",
        "Accuracy Difference",
        "Bootstrap Mean Difference",
        "Bootstrap SE",
        "CI Low 95%",
        "CI High 95%",
    ]

    for (name, task), expected in EXPECTED_BOOT.items():
        r = one_row(df, Role=name, Task=task)
        for col, value in zip(cols, expected):
            eq(f"Table 7 {name}/{task} {col}", r[col], value)


def verify_aav():
    df = read_csv(OUT / "table7_aav_association.csv")
    boot = read_csv(
        ROOT / "outputs" / "aav_external_validation"
        / "02_bootstrap_delta_spearman.csv"
    )

    cols = [
        "AAV-WAR rho", "AAV-Composite rho",
        "Difference", "CI Low 95%", "CI High 95%",
    ]

    for name, expected in EXPECTED_AAV.items():
        r = one_row(df, Role=name)
        b = one_row(boot, role=name)

        for col, value in zip(cols, expected):
            eq(f"Table 8 {name} {col}", r[col], value)

        eq(
            f"AAV Bootstrap repetitions {name}",
            b["bootstrap_reps_valid"], 10000, 0
        )
        check(
            f"AAV Bootstrap unit {name}",
            b["bootstrap_unit"] == "player_key",
        )


def verify_sensitivity():
    df = read_csv(OUT / "table8_sensitivity_recomputed.csv")
    grid = read_csv(
        OUT / "weight_grid" / "weight_grid_metrics.csv"
    )

    pca = {
        "Hitter": .858,
        "Starting Pitcher": .847,
        "Relief Pitcher": .793,
    }

    for name, value in pca.items():
        r = one_row(df, Role=name)
        eq(f"PCA {name}", r["PC1 Variance Explained"], value)

    c2 = [
        col for col in df.columns
        if col.startswith("2-class Label Agreement")
    ]
    c3 = [
        col for col in df.columns
        if col.startswith("3-class Label Agreement")
    ]

    check("2-class agreement columns", len(c2) == 2)
    check("3-class agreement columns", len(c3) == 2)

    eq(
        "2-class agreement minimum",
        df[c2].min().min(), .971,
    )
    eq(
        "2-class agreement maximum",
        df[c2].max().max(), .991,
    )
    eq(
        "3-class agreement minimum",
        df[c3].min().min(), .942,
    )
    eq(
        "3-class agreement maximum",
        df[c3].max().max(), .963,
    )

    eq("Weight-grid conditions", len(grid), 33, 0)

    for role, (name, _, _, _, ntest) in ROLES.items():
        sub = grid[grid["role"].eq(role)]
        eq(f"Grid rows {role}", len(sub), 11, 0)

        r = sub[np.isclose(sub["war_weight"], 0.5)]
        if len(r) != 1:
            raise ValueError(f"Missing 0.5 weight for {role}")
        r = r.iloc[0]

        eq(f"Grid baseline N {role}", r["n_test"], ntest, 0)
        eq(
            f"Grid baseline Accuracy {role}",
            r["test_accuracy"], EXPECTED_2[name][2],
        )
        eq(
            f"Grid baseline High F1 {role}",
            r["high_f1"], EXPECTED_2[name][5],
        )


def verify_figures():
    df = read_csv(CONTRIB / "all_model_contributions.csv")

    for role in ROLES:
        for task in ["2class", "3class"]:
            expected_model = EXPECTED_MODELS[
                (role, "composite", task)
            ]

            sub = df[
                df["role"].eq(role)
                & df["task"].eq(task)
            ]

            check(
                f"Contribution model {role}/{task}",
                len(sub) > 0
                and set(sub["model"]) == {expected_model},
                f"expected={expected_model}",
            )

            if expected_model == "rf" and len(sub):
                eq(
                    f"RF importance sum {role}/{task}",
                    sub["value"].sum(), 1.0, 3
                )

    for name in [
        "figure5_hitter_model_contributions.png",
        "figure6_pitcher_model_contributions.png",
    ]:
        path = CONTRIB / name
        check(
            f"Figure exists: {name}",
            path.is_file() and path.stat().st_size > 0,
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    sections = [
        ("Cohorts", verify_cohorts),
        ("Cutoffs", verify_cutoffs),
        ("Model selection", verify_model_selection),
        ("Classification", verify_classification),
        ("Bootstrap", verify_bootstrap),
        ("AAV", verify_aav),
        ("Sensitivity", verify_sensitivity),
        ("Figure contributions", verify_figures),
    ]

    for name, fn in sections:
        section(name, fn)

    failures = [
        (name, detail)
        for name, passed, detail in results
        if not passed
    ]

    passed = len(results) - len(failures)
    status = "PASS" if not failures else "FAIL"

    report = [
        f"# Reproduction validation: {status}",
        "",
        f"- Passed: {passed}",
        f"- Failed: {len(failures)}",
        f"- Total checks: {len(results)}",
        "",
        "## Validation results",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]

    for name, ok, detail in results:
        safe_detail = detail.replace("|", "/")
        report.append(
            f"| {name} | {'PASS' if ok else 'FAIL'} "
            f"| {safe_detail} |"
        )

    report_path = OUT / "REPRODUCTION_VALIDATION.md"
    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"\nReproduction validation: {status}")
    print(f"Passed: {passed}, Failed: {len(failures)}")
    print(f"Report: {report_path}")

    for name, detail in failures:
        print(f"[FAIL] {name}: {detail}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
