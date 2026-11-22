
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(module: str) -> None:
    print(f"\n>>> Running {module}", flush=True)

    subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce MLB FA paper results"
    )

    parser.add_argument(
        "--rebuild-cohort",
        action="store_true",
        help="Rebuild processed data from raw sources",
    )

    args = parser.parse_args()

    # 1. Data construction
    if args.rebuild_cohort:
        run("src.preprocessing.build_analysis_cohort_v2")

    # 2. Main prediction models
    run("src.modeling.run_temporal_final_models")

    # 3. AAV association
    run("src.evaluation.aav_external_validation")

    # 4. Model contribution figures
    run("src.plot_final_model_contributions")

    # 5. Composite distribution
    run("src.visualization.figure1")

    # 6. Role-specific cutoff characterization
    run("src.paper.characterize_current_cutoffs")

    # 7. Paper tables and paired bootstrap
    run("src.paper.build_paper_tables")

    # 8. Binary + three-class weight sensitivity
    run("src.paper.sensitivity_analysis_2and3class")

    # 9. Extended weight-grid evaluation
    run("src.paper.weight_grid_sensitivity_2class")

    # 10. Paper figures
    run("src.paper.make_paper_figures")

    # 11. Reproduction validation
    run("src.paper.validate_reproduction")

    print("\nPaper reproduction pipeline completed.")


if __name__ == "__main__":
    main()
