from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.paper.composite_vs_war_additional_analysis_paper import (
    bootstrap_accuracy_uncertainty,
)

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / 'outputs' / 'temporal_accuracy_final' / 'temporal_metrics.csv'
PREDICTIONS = ROOT / 'outputs' / 'temporal_accuracy_final' / 'temporal_predictions.csv'
SPEC_PATH = ROOT / 'config' / 'paper_model_spec.json'
OUT = ROOT / 'outputs' / 'paper'

ROLE_LABEL = {'hitter': 'Hitter', 'sp': 'Starting Pitcher', 'rp': 'Relief Pitcher'}


def load_spec():
    return json.loads(SPEC_PATH.read_text(encoding='utf-8'))

def select_paper_models(metrics):
    """
    Select manuscript models using validation data only.

    Candidates: Logistic Regression and Random Forest.
    XGBoost remains an additional comparison model.
    """
    valid = metrics.loc[
        (metrics["split"] == "valid")
        & (metrics["model"].isin(["logistic", "rf"]))
    ].copy()

    valid = valid.sort_values(
        [
            "role", "outcome", "task",
            "accuracy", "balanced_accuracy", "model"
        ],
        ascending=[
            True, True, True,
            False, False, True
        ],
    )

    selected = valid.drop_duplicates(
        ["role", "outcome", "task"]
    ).copy()

    if len(selected) != 12:
        raise RuntimeError(
            f"Expected 12 selected models, got {len(selected)}"
        )

    # Existing paper configuration is for verification only.
    expected = load_spec()

    for row in selected.itertuples(index=False):
        expected_model = expected[row.role][row.task][row.outcome]

        if row.model != expected_model:
            raise RuntimeError(
                f"Paper model mismatch: "
                f"{row.role}/{row.outcome}/{row.task}: "
                f"selected={row.model}, expected={expected_model}"
            )

    selected.to_csv(
        OUT / "paper_selected_models.csv",
        index=False,
    )

    return {
        (row.role, row.outcome, row.task): row.model
        for row in selected.itertuples(index=False)
    }


def selected_row(metrics, role, outcome, task, model):
    q = metrics[
        (metrics.role == role) & (metrics.outcome == outcome) &
        (metrics.task == task) & (metrics.model == model) &
        (metrics.split == 'test')
    ]
    if len(q) != 1:
        raise ValueError(f'Expected one metric row: {role=} {outcome=} {task=} {model=}; got {len(q)}')
    return q.iloc[0]


def paired_bootstrap_stats(
    pred, role, task, comp_model, war_model,
    n_boot=10000, seed=42
):
    c = pred[
        (pred.role == role)
        & (pred.outcome == "composite")
        & (pred.task == task)
        & (pred.model == comp_model)
        & (pred.split == "test")
    ].copy()

    w = pred[
        (pred.role == role)
        & (pred.outcome == "war")
        & (pred.task == task)
        & (pred.model == war_model)
        & (pred.split == "test")
    ].copy()

    keys = ["player_key", "fa_year"]

    z = c[keys + ["y_true", "y_pred"]].merge(
        w[keys + ["y_true", "y_pred"]],
        on=keys,
        suffixes=("_composite", "_war"),
        validate="one_to_one",
    ).sort_values(keys).reset_index(drop=True)

    c_ok = (
        z["y_true_composite"].to_numpy()
        == z["y_pred_composite"].to_numpy()
    ).astype(float)

    w_ok = (
        z["y_true_war"].to_numpy()
        == z["y_pred_war"].to_numpy()
    ).astype(float)

    # Keep the same RNG as the existing manuscript pipeline.
    rng = np.random.RandomState(seed)

    n = len(z)
    diffs = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        diffs[i] = c_ok[idx].mean() - w_ok[idx].mean()

    return {
        "mean": float(diffs.mean()),
        "se": float(diffs.std(ddof=1)),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS)
    pred = pd.read_csv(PREDICTIONS)
    selected = select_paper_models(metrics)
    bootstrap_table = bootstrap_accuracy_uncertainty()

    # Table 4: selected three-class Composite models
    rows = []
    for role in ['hitter', 'sp', 'rp']:
        model = selected[(role, "composite", "3class")]
        r = selected_row(metrics, role, 'composite', '3class', model)
        rows.append({
            'Role': ROLE_LABEL[role], 'Selected model': model,
            'Accuracy': r.accuracy, 'Macro-F1': r.f1_macro,
            'High Recall': r.recall_high, 'High F1': r.f1_high,
        })
    pd.DataFrame(rows).to_csv(OUT/'table4_three_class_composite.csv', index=False)

    # Table 5: selected binary Composite models
    rows = []
    for role in ['hitter', 'sp', 'rp']:
        model = selected[(role, "composite", "2class")]
        r = selected_row(metrics, role, 'composite', '2class', model)
        rows.append({
            'Role': ROLE_LABEL[role], 'Selected model': model,
            'Threshold': r.threshold, 'Accuracy': r.accuracy,
            'Precision': r.precision, 'Recall': r.recall,
            'F1': r.f1, 'PR-AUC': r.pr_auc,
        })
    pd.DataFrame(rows).to_csv(OUT/'table5_binary_composite.csv', index=False)

    # Table 6: Composite vs WAR-only + paired bootstrap CI
    rows = []
    for task in ['2class', '3class']:
        for role in ['hitter', 'sp', 'rp']:
            cm = selected[(role, "composite", task)]
            wm = selected[(role, "war", task)]
            cr = selected_row(metrics, role, 'composite', task, cm)
            wr = selected_row(metrics, role, 'war', task, wm)
            matched = bootstrap_table.loc[
                (bootstrap_table["Role"] == ROLE_LABEL[role])
                & (bootstrap_table["Task"] == task)
            ]

            if len(matched) != 1:
                raise ValueError(
                    f"Invalid bootstrap result: {role}, {task}"
                )

            boot = matched.iloc[0]

            if (
                boot["Composite_Model"] != cm
                or boot["WAR_Model"] != wm
            ):
                raise ValueError(
                    f"Bootstrap model mismatch: {role}, {task}"
                )
            rows.append({
                'Role': ROLE_LABEL[role], 'Task': task,
                'Composite model': cm, 'WAR-only model': wm,
                'Composite Accuracy': round(float(cr.accuracy), 3),
                'WAR-only Accuracy': round(float(wr.accuracy), 3),
                # The manuscript reports the difference from the displayed 3-decimal accuracies.
                'Accuracy Difference': round(float(cr.accuracy), 3) - round(float(wr.accuracy), 3),
                'Accuracy Difference Raw': float(cr.accuracy - wr.accuracy),
                "Bootstrap Mean Difference": round(
                    float(boot["Bootstrap_Mean_Difference"]), 3
                ),
                "Bootstrap SE": round(
                    float(boot["Bootstrap_SE"]), 3
                ),
                "CI Low 95%": round(
                    float(boot["CI_2.5%"]), 3
                ),
                "CI High 95%": round(
                    float(boot["CI_97.5%"]), 3
                ),

                "Bootstrap Mean Difference": round(
                    float(boot["Bootstrap_Mean_Difference"]), 3
                ),
                "Bootstrap SE": round(
                    float(boot["Bootstrap_SE"]), 3
                ),
                "CI Low 95%": round(
                    float(boot["CI_2.5%"]), 3
                ),
                "CI High 95%": round(
                    float(boot["CI_97.5%"]), 3
                ),

            })
    pd.DataFrame(rows).to_csv(OUT/'table6_composite_vs_war_bootstrap.csv', index=False)

    # Table 7: AAV association with WAR and Composite
    corr_path = ROOT / "outputs" / "aav_external_validation" / "01_spearman_aav_association.csv"
    boot_path = ROOT / "outputs" / "aav_external_validation" / "02_bootstrap_delta_spearman.csv"
    if corr_path.exists() and boot_path.exists():
        corr = pd.read_csv(corr_path)
        boot = pd.read_csv(boot_path)
        table7 = []
        for role_name in ["Hitter", "Starting Pitcher", "Relief Pitcher"]:
            c = corr[corr["role"].eq(role_name)].set_index("metric")
            b = boot[boot["role"].eq(role_name)].iloc[0]
            table7.append({
                "Role": role_name,
                "AAV-WAR rho": round(float(c.loc["WAR", "spearman_rho_with_AAV"]), 3),
                "AAV-Composite rho": round(float(c.loc["Composite", "spearman_rho_with_AAV"]), 3),
                "Difference": round(float(b["delta_rho_Composite_minus_WAR"]), 3),
                "CI Low 95%": round(float(b["ci_low_2.5"]), 3),
                "CI High 95%": round(float(b["ci_high_97.5"]), 3),
            })
        pd.DataFrame(table7).to_csv(OUT / "table7_aav_association.csv", index=False)

    print('Saved paper tables to', OUT)

if __name__ == '__main__':
    main()
