"""Select role-specific High cutoffs without touching the 2020-2021 test set.

What this script changes
------------------------
1. Removes the old hard-coded Hitter=40%, SP/RP=20% rule.
2. Searches a pre-declared candidate range using ONLY validation performance.
3. Uses Composite validation High-class F1 as the primary criterion for the
   role-specific percentile cutoff.
4. After the role cutoff is locked, selects the best LR/RF/XGB model separately
   for Composite and WAR-only on validation F1 under the SAME top proportion.
5. Locks the probability threshold selected on validation and evaluates once on
   the untouched 2020-2021 test set.

Default candidate range: top 15%, 20%, 25%, 30%, 35%, 40%, 45%.
You can change it from the command line, e.g.:
    python src/modeling/run_optimized_2class.py --cutoffs 0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from optimized_2class_common import (
    OUTPUT,
    ROLE_CONFIG,
    ROLE_LABELS,
    add_binary_labels,
    binary_metrics,
    candidate_models,
    choose_best_validation_row,
    fit_pipeline,
    load_role_frame,
    prepare_scores,
    select_probability_threshold,
)


def parse_cutoffs(text: str) -> list[float]:
    values = sorted({round(float(x.strip()), 6) for x in text.split(",") if x.strip()})
    if not values:
        raise ValueError("At least one cutoff candidate is required")
    bad = [x for x in values if not (0 < x < 0.5)]
    if bad:
        raise ValueError(f"All candidate top ratios must be in (0, 0.5): {bad}")
    return values


def evaluate_validation_configuration(
    labeled: pd.DataFrame,
    role: str,
    outcome: str,
    top_ratio: float,
) -> tuple[list[dict], dict[tuple[str, str], object]]:
    """Fit all candidate model families and score them on validation data."""
    cfg = ROLE_CONFIG[role]
    features = cfg["features"]
    train = labeled[labeled["split"].eq("train")].copy()
    valid = labeled[labeled["split"].eq("valid")].copy()
    target = f"{outcome}_2class"

    if train[target].nunique() < 2:
        raise ValueError(f"{role}/{outcome}/{top_ratio}: training labels have one class")
    if valid[target].sum() == 0:
        # We keep the candidate but it cannot meaningfully optimize High F1.
        print(f"[warning] {role}/{outcome}/top={top_ratio:.2f}: validation has zero High rows")

    rows: list[dict] = []
    pipelines: dict[tuple[str, str], object] = {}
    for model_name, estimator in candidate_models(train[target]).items():
        pipe = fit_pipeline(train, features, target, estimator)
        probability = pipe.predict_proba(valid[features])[:, 1]
        threshold, _ = select_probability_threshold(valid[target], probability)
        metrics, _ = binary_metrics(valid[target], probability, threshold)
        row = {
            "role": role,
            "role_label": ROLE_LABELS[role],
            "outcome": outcome,
            "top_ratio": float(top_ratio),
            "model": model_name,
            "split": "valid",
            "train_n": int(len(train)),
            "valid_n": int(len(valid)),
            "train_positive_count": int(train[target].sum()),
            "valid_positive_count": int(valid[target].sum()),
            **metrics,
        }
        rows.append(row)
        pipelines[(outcome, model_name)] = pipe
    return rows, pipelines


def run(cutoffs: list[float]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    cutoff_search_rows: list[dict] = []
    selected_validation_rows: list[dict] = []
    final_test_rows: list[dict] = []
    final_prediction_frames: list[pd.DataFrame] = []
    final_confusion_rows: list[dict] = []
    class_count_rows: list[dict] = []
    selected_config: dict = {
        "selection_rule": {
            "role_cutoff": "maximize Composite validation High-class F1; tie-break by balanced accuracy, PR-AUC, recall, validation High count, then simpler model",
            "model": "within the selected role cutoff, maximize validation High-class F1 separately for Composite and WAR-only with the same tie-break rule",
            "probability_threshold": "maximize validation High-class F1; tie-break by balanced accuracy, recall, then closeness to 0.50",
            "test_usage": "2020-2021 test set is not used for cutoff/model/threshold selection",
        },
        "candidate_top_ratios": cutoffs,
        "roles": {},
    }
    label_params: dict[str, dict] = {}

    for role, cfg in ROLE_CONFIG.items():
        print(f"\n{'=' * 78}\n{ROLE_LABELS[role]}\n{'=' * 78}")
        raw = load_role_frame(role)
        scored, standardization = prepare_scores(raw, cfg)

        # ------------------------------------------------------------
        # 1) ROLE-SPECIFIC CUTOFF SEARCH — COMPOSITE ONLY, VALIDATION ONLY
        # ------------------------------------------------------------
        role_search_rows: list[dict] = []
        for top_ratio in cutoffs:
            labeled, cuts = add_binary_labels(scored, cfg, top_ratio)
            rows, _ = evaluate_validation_configuration(
                labeled=labeled,
                role=role,
                outcome="composite",
                top_ratio=top_ratio,
            )
            for row in rows:
                row["composite_label_cut"] = cuts["composite_cut"]
                role_search_rows.append(row)
                cutoff_search_rows.append(row)

        role_search = pd.DataFrame(role_search_rows)
        best_cutoff_row = choose_best_validation_row(role_search)
        selected_top = float(best_cutoff_row["top_ratio"])
        print(
            f"Selected top ratio = {selected_top:.0%} | "
            f"Composite validation F1={best_cutoff_row['f1']:.3f} | "
            f"model={best_cutoff_row['model']}"
        )

        # ------------------------------------------------------------
        # 2) LOCK THE ROLE CUTOFF; SELECT MODEL/PROBABILITY THRESHOLD
        #    SEPARATELY FOR COMPOSITE AND WAR-ONLY ON VALIDATION ONLY
        # ------------------------------------------------------------
        labeled, selected_cuts = add_binary_labels(scored, cfg, selected_top)
        label_params[role] = {
            **standardization,
            **selected_cuts,
        }

        role_config_out: dict = {
            "selected_top_ratio": selected_top,
            "selected_top_percent": 100.0 * selected_top,
            "composite_cut": selected_cuts["composite_cut"],
            "war_cut": selected_cuts["war_cut"],
            "outcomes": {},
        }

        # Save class counts after the selected cutoff is locked.
        for split_name, frame in labeled.groupby("split"):
            for outcome in ("composite", "war"):
                target = f"{outcome}_2class"
                class_count_rows.append(
                    {
                        "role": role,
                        "role_label": ROLE_LABELS[role],
                        "outcome": outcome,
                        "top_ratio": selected_top,
                        "split": split_name,
                        "n": int(len(frame)),
                        "n_players": int(frame["player_key"].nunique()),
                        "high_count": int(frame[target].sum()),
                        "high_rate": float(frame[target].mean()),
                    }
                )

        for outcome in ("composite", "war"):
            validation_rows, _ = evaluate_validation_configuration(
                labeled=labeled,
                role=role,
                outcome=outcome,
                top_ratio=selected_top,
            )
            valid_table = pd.DataFrame(validation_rows)
            best_model_row = choose_best_validation_row(valid_table)
            selected_validation_rows.extend(validation_rows)

            model_name = str(best_model_row["model"])
            threshold = float(best_model_row["threshold"])
            target = f"{outcome}_2class"
            train = labeled[labeled["split"].eq("train")].copy()
            valid = labeled[labeled["split"].eq("valid")].copy()
            test = labeled[labeled["split"].eq("test")].copy()
            features = cfg["features"]

            # Refit the selected model on TRAIN only, exactly matching the paper's
            # train/validation/test protocol. Validation is used for selection,
            # not merged into training before the final test.
            estimator = candidate_models(train[target])[model_name]
            pipe = fit_pipeline(train, features, target, estimator)

            valid_probability = pipe.predict_proba(valid[features])[:, 1]
            # Recalculate the threshold from the selected model's validation
            # probabilities to avoid relying on a stale table value.
            threshold, threshold_table = select_probability_threshold(
                valid[target], valid_probability
            )
            valid_metrics, valid_pred = binary_metrics(
                valid[target], valid_probability, threshold
            )

            test_probability = pipe.predict_proba(test[features])[:, 1]
            test_metrics, test_pred = binary_metrics(
                test[target], test_probability, threshold
            )

            final_test_rows.append(
                {
                    "role": role,
                    "role_label": ROLE_LABELS[role],
                    "outcome": outcome,
                    "task": "2class",
                    "selected_top_ratio": selected_top,
                    "selected_model": model_name,
                    "split": "test",
                    "n": int(len(test)),
                    "positive_count": int(test[target].sum()),
                    **test_metrics,
                }
            )

            role_config_out["outcomes"][outcome] = {
                "model": model_name,
                "probability_threshold": threshold,
                "validation_metrics": valid_metrics,
                "test_metrics": test_metrics,
            }

            for split_name, frame, probability, prediction in (
                ("valid", valid, valid_probability, valid_pred),
                ("test", test, test_probability, test_pred),
            ):
                pred_frame = frame[["contract_row_id", "player_key", "fa_year"]].reset_index(drop=True).copy()
                # Player name is useful for downstream case examples if present.
                if "player" in frame.columns:
                    pred_frame["player"] = frame["player"].reset_index(drop=True)
                pred_frame["role"] = role
                pred_frame["role_label"] = ROLE_LABELS[role]
                pred_frame["outcome"] = outcome
                pred_frame["task"] = "2class"
                pred_frame["model"] = model_name
                pred_frame["selected_top_ratio"] = selected_top
                pred_frame["split"] = split_name
                pred_frame["threshold"] = threshold
                pred_frame["y_true"] = frame[target].to_numpy()
                pred_frame["y_pred"] = prediction
                pred_frame["probability_high"] = probability
                final_prediction_frames.append(pred_frame)

                matrix = confusion_matrix(frame[target], prediction, labels=[0, 1])
                for true_class in (0, 1):
                    for predicted_class in (0, 1):
                        final_confusion_rows.append(
                            {
                                "role": role,
                                "outcome": outcome,
                                "split": split_name,
                                "model": model_name,
                                "top_ratio": selected_top,
                                "true_class": true_class,
                                "predicted_class": predicted_class,
                                "count": int(matrix[true_class, predicted_class]),
                            }
                        )

            threshold_table.assign(
                role=role,
                outcome=outcome,
                model=model_name,
                top_ratio=selected_top,
            ).to_csv(
                OUTPUT / f"threshold_search_{role}_{outcome}.csv",
                index=False,
                encoding="utf-8-sig",
            )

        selected_config["roles"][role] = role_config_out

    # ----------------------------------------------------------------
    # WRITE ALL OUTPUTS
    # ----------------------------------------------------------------
    pd.DataFrame(cutoff_search_rows).to_csv(
        OUTPUT / "cutoff_search_validation.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(selected_validation_rows).to_csv(
        OUTPUT / "selected_cutoff_model_search_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(final_test_rows).to_csv(
        OUTPUT / "final_test_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(final_prediction_frames, ignore_index=True).to_csv(
        OUTPUT / "final_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(final_confusion_rows).to_csv(
        OUTPUT / "final_confusion_matrices.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(class_count_rows).to_csv(
        OUTPUT / "selected_cutoff_class_counts.csv", index=False, encoding="utf-8-sig"
    )

    with (OUTPUT / "selected_role_config.json").open("w", encoding="utf-8") as f:
        json.dump(selected_config, f, ensure_ascii=False, indent=2)
    with (OUTPUT / "selected_label_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(label_params, f, ensure_ascii=False, indent=2)

    print("\nFinal selected configurations")
    print(pd.DataFrame(final_test_rows)[
        [
            "role_label",
            "outcome",
            "selected_top_ratio",
            "selected_model",
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "pr_auc",
        ]
    ].to_string(index=False))
    print(f"\nSaved to: {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cutoffs",
        default="0.15,0.20,0.25,0.30,0.35,0.40,0.45",
        help="Comma-separated High top ratios. Test data is never used to choose among them.",
    )
    args = parser.parse_args()
    run(parse_cutoffs(args.cutoffs))


if __name__ == "__main__":
    main()
