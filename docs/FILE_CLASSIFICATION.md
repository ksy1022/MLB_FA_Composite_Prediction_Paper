# 파일 분류 기준 — 최종 논문 재현 저장소

2026-11-22 정리 작업 기준. 과거 실험물은 프로젝트 외부의 로컬 백업으로 분리했으며, Git에서 삭제 변경으로 등록한 후 최종 커밋하기 전 상태입니다.

## 1. 공식 파이프라인: 반드시 유지

### 설정·실행

- `run_all.py` — **공식 단일 실행 진입점** (`python run_all.py`)
- `config/paper_model_spec.json` — 논문 표에 보고할 대표 모델의 명시적 스펙
- `requirements.txt`, `.gitignore`, `README.md`
- `docs/REPRODUCIBILITY.md`, `docs/FILE_CLASSIFICATION.md`

### 필수 데이터

- `data/raw/` — 원시 데이터에서 재생성할 때 필요
- `data/processed/` — 정제된 계약 정보 등 전처리 입력
- `data/processed_v2/` — 기본 파이프라인이 읽는 최종 코호트

### Python 코드 (현재 `src/` 13개)

| 파일 | 용도 |
|---|---|
| `src/preprocessing/build_analysis_cohort_v2.py` | 원시 데이터에서 코호트 재생성 |
| `src/modeling/run_temporal_final_models.py` | 후보 모델 학습, 시간순 검증·시험 |
| `src/evaluation/aav_external_validation.py` | 계약 AAV 분석 |
| `src/plot_final_model_contributions.py` | Figure 5·6 |
| `src/visualization/figure1.py` | Figure 1 |
| `src/paper/characterize_current_cutoffs.py` | 역할별 High 컷오프 기술통계 |
| `src/paper/build_paper_tables.py` | 논문 표·Bootstrap |
| `src/paper/composite_vs_war_additional_analysis_paper.py` | 표 생성에서 사용하는 Bootstrap 계산 함수도 포함; 단순 삭제 금지 |
| `src/paper/sensitivity_analysis_2and3class.py` | PCA 및 2/3-class 가중치 민감도 |
| `src/paper/weight_grid_sensitivity_2class.py` | 2-class 가중치 그리드 실험 |
| `src/paper/validate_weight_grid_2class.py` | 그리드 33조건 재계산 검증 |
| `src/paper/make_paper_figures.py` | 최종 논문 비교 그림 |
| `src/paper/validate_reproduction.py` | 논문 수치 재현 검사 |

`composite_vs_war_additional_analysis_paper.py`는 파일명에 `additional`이 있어도 현재 `build_paper_tables.py`에서 쓰는 계산 함수의 출처이므로 참조를 옮기기 전에는 삭제하지 않습니다.

## 2. 공식 출력: 현재 유지

- `outputs/temporal_accuracy_final/` — 핵심 예측·모델 지표·라벨 파라미터
- `outputs/aav_external_validation/` — AAV 분석 결과
- `outputs/final_model_contributions/` — 변수 기여도 및 Figure 5·6
- `outputs/paper_figures/` — Figure 1
- `outputs/paper/` — 표·그림·검증·민감도·컷오프 기술통계
- `outputs/reference/` — 기준 스냅샷; 코드가 직접 갱신하는 폴더와 구분

**참고:** `outputs/paper/` 내부에는 현재 실행에서 갱신되는 파일과 과거 분석 파일이 함께 있을 수 있습니다. 내부 정리는 개별 파일의 생산자/소비자 경로를 확인한 후 진행합니다. `outputs/` 전체를 무조건 `.gitignore`에 넣거나 삭제하지 않습니다.

### 추가로 점검할 파일

- `outputs/final_model_contributions/rp_2class_logistic_coefficients.csv` — 최종 RP 2-class 대표모델은 RF이므로 과거 잔존 산출물인지 확인
- `outputs/final_model_contributions/rp_2class_rf_importance.csv` — 최신 RF 결과; 유지 후보
- `outputs/paper/paper_selected_models.csv` — 대표 모델 스펙/선정 과정 확인용; 유지 후보
- `outputs/paper/figure9_construct_validity_weight_sensitivity.png` — 현재 `run_all.py`에서는 재생성하지 않는 과거 Figure 9 파일

## 3. 연구 이력으로 분리한 파일

### 과거 Python 코드

- 기존 `src/` 실험 코드 **21개**는 프로젝트 밖 `MLB_FA_reproduction_backup/code_snapshots/.../archived_experiments/`로 이동
- 기존 `legacy_related/` 코드는 `MLB_FA_reproduction_backup/legacy_related/`로 이동
- 이들은 공식 `run_all.py`에 포함되지 않으며, 논문 코드 재현을 위해 실행할 필요가 없음

### 과거 실험 출력 폴더 5개

- `outputs/optimized_2class/`
- `outputs/fixed_tertile_2class/`
- `outputs/cutoff_sensitivity_203040/`
- `outputs/gmm3_structure_validation/`
- `outputs/composite_distribution_structure/`

이 폴더들은 `MLB_FA_reproduction_backup/archived_outputs/`에 보관한 상태입니다. Git에서는 백업으로 이동한 파일들이 삭제(`D`) 변경으로 표시되는 것이 정상입니다.

### 임시 재생성 점검

- `outputs/rebuild_check/` — 원시 데이터→코호트 일치 검증 결과; 공식 출력 경로에서는 제외하고 로컬 백업에 보관

## 4. Git 관리 원칙

1. `.gitignore`에 과거 출력 폴더 및 `outputs/rebuild_check/`를 등록합니다.
2. 이미 Git에서 추적하던 파일은 `.gitignore`만으로 추적이 해제되지 않으므로, **백업 후 이동한 파일의 삭제 변경**을 확인하여 스테이징합니다.
3. `data/processed_v2/`, 공식 코드, `config/`, `outputs/reference/`를 무심코 삭제하지 않습니다.
4. `run_all.py`가 변경시킨 대량의 CSV/PNG는 **논문에 필요한 출력 또는 기준 스냅샷인지 검토한 후** 커밋 대상을 선택합니다.
5. GitHub 업로드 전 `git diff --cached --check`, `python run_all.py` 및 재현성 검증을 수행합니다.
6. 원천 Baseball-Reference/Spotrac 자료의 사용·재배포 조건은 별도로 확인합니다.

## 5. 검증 결과와 한계

2026-11-22 로컬 실행에서 `run_all.py` 및 가중치 검증이 성공했습니다(재현성 217/217 PASS, 33조건 확인). 이는 **현재 코드와 검사 대상 수치의 일치**를 입증하지만, 대표 모델의 선정 근거나 원고의 모든 통계적 표현을 자동으로 검증하는 것은 아닙니다. 모델 스펙과 자동 선정 결과의 차이는 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)에 기록합니다.
