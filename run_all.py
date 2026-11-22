from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def run(rel):
    print(f'\n>>> {rel}')
    subprocess.run([sys.executable, str(ROOT/rel)], cwd=ROOT, check=True)

def main():
    p=argparse.ArgumentParser(description='Reproduce the final MLB FA paper outputs.')
    p.add_argument('--rebuild-cohort',action='store_true',help='Rebuild processed_v2 from raw Baseball-Reference + cleaned Spotrac data first.')
    a=p.parse_args()
    if a.rebuild_cohort: run('src/preprocessing/build_analysis_cohort_v2.py')
    run('src/modeling/run_temporal_final_models.py')
    run('src/evaluation/aav_external_validation.py')
    run('src/plot_final_model_contributions.py')
    run('src/visualization/figure1.py')
    run('src/paper/build_paper_tables.py')
    run('src/paper/sensitivity_analysis.py')
    run('src/paper/make_paper_figures.py')
    run('src/paper/validate_reproduction.py')
    print('\nDone. Final manuscript-aligned outputs: outputs/paper/')

if __name__=='__main__': main()
