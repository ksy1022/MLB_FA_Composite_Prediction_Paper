# MLB Free Agent Performance Prediction — Paper Reproduction

**Paper:** *Predicting the Performance of MLB Free Agents Upon Signing: A Machine Learning Approach*
**연구:** MLB 자유계약선수의 계약 후 성과 예측: 머신러닝 기반 접근

이 저장소는 **계약 당시 이용 가능한 정보**로 MLB 자유계약선수(FA)의 **계약 후 3시즌 상대적 성과**를 분류한 연구의 데이터 처리, 모델 학습, 표·그림 생성 및 재현성 검증 코드를 제공합니다. 최종 논문 재현용 진입점은 프로젝트 루트의 **`run_all.py` 한 개**입니다.

## 1. 연구 개요

- 계약 연도: **2012–2021년**, FA 계약 관측치 **1,162건**
- 역할별 표본: 타자 **534건**, 선발투수 **216건**, 불펜투수 **412건**
- 입력: 계약 이전 성과·출전 정보, 연령, 포지션, 계약정보(AAV·계약기간 등)
- 결과: 계약 후 3시즌 WAR와 경기 참여량(PA 또는 IP)을 바탕으로 정의한 상대적 성과집단
- 비교 대상: **WAR-only** vs. **Composite Score**
- 모델 후보: Logistic Regression, Random Forest, XGBoost

### 시간순 분할

| 구간 | FA 계약 연도 | 용도 |
|---|---|---|
| Train | 2012–2017 | 전처리 통계량·라벨 경계 추정 및 모델 학습 |
| Validation | 2018–2019 | 예측 임계값 및 모델 평가·선택 절차 |
| Test | 2020–2021 | 최종 시계열 외부구간 평가 |

라벨 계산에 쓰는 결측값 대치·표준화·컷오프는 **학습 구간을 기준**으로 추정합니다. 상세한 선정·평가 절차는 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)를 참고하세요.

## 2. 성과지표와 분류 문제

**Composite Score**는 WAR와 실제 경기 참여도를 함께 반영하는 보완적 기준입니다.

\[
\mathrm{Composite}=0.5\,z(\mathrm{WAR}_{3y})+0.5\,z(\mathrm{PA/IP}_{3y})
\]

- 타자: WAR + PA(타석 수)
- 선발·불펜투수: WAR + IP(투구 이닝)
- **2-class:** 타자는 상위 40%를 `High`, 선발·불펜은 상위 20%를 `High`로 정의; 나머지는 `Other`
- **3-class:** 역할별 성과분포의 약 1/3 및 2/3 경계를 이용하여 `Low` / `Middle` / `High` 구분

각 집단은 **절대적인 계약 성공·실패가 아니라 같은 역할의 FA 계약 간 상대적 성과**입니다. PA/IP는 능력만이 아닌 부상, 팀 사정, 기용 방식 등의 영향도 받습니다. 코드 내부 일부 계산은 두 z-score의 합을 사용하므로 0.5 배의 Composite와 수치 척도는 다를 수 있지만, 순위 기반 분류는 동일합니다.

## 3. 실행 환경

`requirements.txt`에 핵심 패키지 버전이 명시돼 있습니다. Windows PowerShell 기준:

```powershell
# 프로젝트 루트에서 실행
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.11을 예로 들었습니다. 이미 작동하는 가상환경이 있다면 다시 만들 필요가 없습니다.

## 4. 재현 실행

### 4.1. 가공된 연구 코호트에서 전체 결과 재현

```powershell
python run_all.py
```

프로젝트 루트에 `data/processed_v2/` 코호트 파일들이 있어야 합니다. 실행은 모델 평가, AAV 분석, 그림, 논문 표, 민감도 및 검증 보고서를 **의존 순서대로** 생성합니다.

### 4.2. 원시 데이터에서 코호트도 재생성

```powershell
python run_all.py --rebuild-cohort
```

이 옵션은 필요한 Baseball-Reference 기록 및 정제된 Spotrac 계약 원천 데이터가 `data/raw/`와 `data/processed/` 아래에 준비된 경우에 사용합니다. 수집 원천과 재배포 허용 여부는 각 제공처 정책을 확인하세요.

### 4.3. 검증만 별도로 실행

```powershell
python -m src.paper.validate_weight_grid_2class
python -m src.paper.validate_reproduction
```

첫 명령은 `outputs/paper/weight_grid/`에 **이미 생성된** 가중치별 결과가 필요합니다. 두 번째 명령도 논문 결과 CSV가 먼저 생성돼 있어야 합니다. 처음 실행할 때는 `python run_all.py`를 권장합니다.

**최근 로컬 실행 기록(2026-11-22):** 논문 자동 검증 **217/217 PASS**, 가중치 **33/33개 조건의 지표·라벨 재검산 PASS**. 이 검증은 생성된 수치와 명시된 체크 조건의 일치를 보여주며, 모든 연구 설계 판단의 통계적 타당성을 보증하지는 않습니다.

## 5. 공식 실행 단계

`run_all.py`는 다음 순서대로 Python 모듈을 실행합니다.

| 단계 | 모듈 | 역할 |
|---:|---|---|
| 선택 | `src.preprocessing.build_analysis_cohort_v2` | `--rebuild-cohort` 시 코호트 생성 |
| 1 | `src.modeling.run_temporal_final_models` | 후보모델 학습·시계열 분할 평가 |
| 2 | `src.evaluation.aav_external_validation` | AAV와 두 성과지표의 연관성 |
| 3 | `src.plot_final_model_contributions` | Figure 5·6 변수 기여도 |
| 4 | `src.visualization.figure1` | Composite 분포 Figure 1 |
| 5 | `src.paper.characterize_current_cutoffs` | 40/20/20 High 기준의 기술 통계 |
| 6 | `src.paper.build_paper_tables` | 논문 표·paired bootstrap |
| 7 | `src.paper.sensitivity_analysis_2and3class` | PCA·2/3-class 라벨 일치율 |
| 8 | `src.paper.weight_grid_sensitivity_2class` | WAR 가중치 0.0–1.0, 0.1 간격 실험 |
| 9 | `src.paper.validate_weight_grid_2class` | 33조건 수치·라벨 검증 |
| 10 | `src.paper.make_paper_figures` | 예측성능·비교 그림 |
| 11 | `src.paper.validate_reproduction` | 논문 결과 자동 검증·보고서 |

**주의:** 과거 README의 `sensitivity_analysis.py`와 `generate_figure9.py`는 최종 실행 경로에서 제외되어 **현재 README에 실행 명령으로 사용하지 않습니다.** 이 파이프라인은 과거 Figure 9를 재생성하지 않습니다.

## 6. 프로젝트 구성

```text
MLB_FA_Composite_Predict_Paper/
├── config/
│   └── paper_model_spec.json
├── data/
│   ├── raw/
│   ├── processed/
│   └── processed_v2/
├── src/
│   ├── preprocessing/build_analysis_cohort_v2.py
│   ├── modeling/run_temporal_final_models.py
│   ├── evaluation/aav_external_validation.py
│   ├── visualization/figure1.py
│   ├── plot_final_model_contributions.py
│   └── paper/
│       ├── build_paper_tables.py
│       ├── characterize_current_cutoffs.py
│       ├── composite_vs_war_additional_analysis_paper.py
│       ├── make_paper_figures.py
│       ├── sensitivity_analysis_2and3class.py
│       ├── weight_grid_sensitivity_2class.py
│       ├── validate_weight_grid_2class.py
│       └── validate_reproduction.py
├── outputs/              # 최종 산출물 및 기준 스냅샷
├── docs/
│   ├── REPRODUCIBILITY.md
│   └── FILE_CLASSIFICATION.md
├── run_all.py
├── requirements.txt
└── .gitignore
```

현재 공식 저장소에서 과거 실험 스크립트와 레거시 코드는 제외했습니다. 연구 과정의 실험 코드는 별도 로컬 백업에 보관하며, 최종 실행에 필요하지 않습니다.

## 7. 주요 출력물

| 위치 | 내용 |
|---|---|
| `outputs/temporal_accuracy_final/` | 모든 후보모델의 평가·예측·라벨 파라미터 |
| `outputs/aav_external_validation/` | Spearman, 부트스트랩 및 회귀 결과 |
| `outputs/final_model_contributions/` | Figure 5·6 및 기여도 CSV |
| `outputs/paper_figures/` | Figure 1 |
| `outputs/paper/` | 논문 표·Figure 2·3·4·7·8·검증 보고서 |
| `outputs/paper/weight_grid/` | 가중치 실험 결과·예측·진단 CSV |
| `outputs/reference/` | 이전 결과의 기준 스냅샷; 자동 실행이 덮어쓰지 않음 |

주요 표 파일은 `table4_three_class_composite.csv`, `table5_binary_composite.csv`, `table6_composite_vs_war_bootstrap.csv`, `table7_aav_association.csv`, `table8_sensitivity_recomputed.csv`입니다. `outputs/paper/REPRODUCTION_VALIDATION.md`에 최신 체크 결과가 기록됩니다. 일부 그림·파일의 번호는 투고 원고 편집판의 표·그림 번호와 다를 수 있으므로 파일명을 기준으로 확인하세요.

## 8. 재현성과 모델 선정에 관한 중요 주의사항

- `run_temporal_final_models.py`는 후보모델 전체를 학습하고 `temporal_best_models.csv`에 **자동 선택 결과**를 기록합니다.
- 논문 표와 기여도 그림에는 **`config/paper_model_spec.json`에 명시된 논문 대표 모델**을 사용합니다. 두 모델 선택 결과가 일부 일치하지 않는다는 점을 공개합니다. **이는 숫자 재현과 별개로 논문 Methods에 선정 규칙을 명확하게 설명해야 하는 문제**입니다.
- 2-class의 Accuracy만 높고 `High-F1=0`일 수 있습니다. 특히 표본이 적은 투수 집단에서는 Accuracy와 함께 High-F1·PR-AUC·다수 클래스 베이스라인을 검토해야 합니다.
- `sensitivity_analysis_2and3class.py`는 **라벨 안정성을 기술적으로 비교**할 목적으로 전체 코호트에서 표준화하여 PCA·라벨 일치율을 계산합니다. 예측 모델의 Train-only 변환과는 분석 목적·추정 구간이 다르므로 두 결과를 같은 전처리 실험으로 혼동하지 않습니다.
- 가중치 0.0 조건의 선발투수 검증 집단에는 High 라벨이 0건이므로 임계값 평가의 해석에 제약이 있습니다. 원래 논문의 중심 분석 범위인 0.1–0.9와 구분해서 해석하세요.
- 검증 `PASS`는 자동 검사 대상의 일치를 의미합니다. 원고에 적힌 신뢰구간(특히 타자 2-class 하한), 모델 선정 설명 및 표·그림 출처는 투고 전 별도로 대조해야 합니다.

자세한 재현 규약 및 미해결 방법론상 유의점: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## 9. 데이터 제공 및 인용

원천 통계는 **Baseball-Reference**, 계약정보는 **Spotrac** 기반입니다. 각 사이트의 출처 표시 및 원천 데이터 재배포 조건을 준수하세요. 이 저장소의 코드가 공개돼 있다고 해서 제3자 데이터의 재배포 권한까지 부여되는 것은 아닙니다.

관련 연구를 인용할 때는 최종 출판판의 저자·학술지·연도·DOI 등 서지정보를 확인해 사용하세요.
