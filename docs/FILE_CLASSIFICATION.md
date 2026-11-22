# 파일 분류 기준

## 1 — Final paper reproduction

현재 논문의 수치와 그림을 직접 재현하는 파일입니다. 이 경로만으로 최종 분석을 실행할 수 있도록 유지합니다.

### Data
- `data/raw/batting/*.csv`
- `data/raw/pitching/*.csv`
- `data/processed/spotrac_fa_2011_2024_clean.csv`
- `data/processed_v2/hitter_cohort_v2.csv`
- `data/processed_v2/sp_cohort_v2.csv`
- `data/processed_v2/rp_cohort_v2.csv`
- `data/processed_v2/pitcher_cohort_v2.csv`

### Code
- `src/preprocessing/build_analysis_cohort_v2.py`
- `src/modeling/run_temporal_final_models.py`
- `src/evaluation/aav_external_validation.py`
- `src/plot_final_model_contributions.py`
- `src/visualization/figure1.py`
- `src/paper/*`
- `run_all.py`

### Outputs kept as reference snapshots
- `outputs/reference/temporal_accuracy_final/`
- `outputs/reference/aav_external_validation/`
- `outputs/reference/final_model_contributions/`
- `outputs/reference/paper_figures/`

These are reference snapshots only. `run_all.py` writes fresh outputs outside `outputs/reference/`.

## 2 — Legacy related

`legacy_related/src/` 아래의 파일입니다. 연구 과정 또는 이전 논문 버전과 관련은 있지만 현재 최종 재현 경로에서 호출하지 않습니다.

Examples:
- 이전 2-class / 3-class modeling code
- cutoff comparison/revalidation code
- 이전 feature-engineering experiments
- WAR baseline comparison prototypes
- 이전 plotting/table scripts
- label-strategy experiments
- earlier cohort-validation utilities

이 폴더는 **참고 및 연구 이력 보존용**이며, 최종 논문 결과를 만들기 위해 실행할 필요가 없습니다.

## 3 — Removed

다음 항목은 GitHub 정리본에서 제거했습니다.

- `.venv/`, `.git/`, `__pycache__/`, `*.pyc`
- WPA 분석 관련 데이터·코드·결과
- actual on-field success 별도 검증
- repeated group regression / regression pilot
- feature-block ablation
- data-driven clustering / composite-profile 확장분석
- VIF 별도 실험
- SHAP 별도 실험
- 로컬 업로드/중복 데이터
- 최종 논문에 사용되지 않는 대량의 구버전 output 폴더
- 최종 cohort 재현에 필요하지 않은 64MB 외부 register-master 복사본

삭제 원칙은 “오래된 파일”이 아니라 **최종 논문 재현에도 필요 없고 연구 이력 보존 가치도 낮은 파일**입니다.
