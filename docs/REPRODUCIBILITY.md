# Reproducibility — MLB FA Composite Prediction Paper

> 이 문서는 정리된 **공식 재현 파이프라인(`run_all.py`)**을 설명합니다. 과거 실험 코드 및 이전 README의 실행 안내를 현재 경로로 사용하지 마세요.

## 1. 재현 단위와 데이터

- 분석단위: 2012–2021년의 FA 계약 관측치(동일 선수의 여러 계약이 있을 수 있음)
- 역할별 코호트: Hitter 534, Starting Pitcher 216, Relief Pitcher 412; 합계 1,162
- Train 2012–2017, Validation 2018–2019, Test 2020–2021
- 기본 실행은 `data/processed_v2/{hitter,sp,rp}_cohort_v2.csv`를 사용
- `python run_all.py --rebuild-cohort`는 원시/정제 계약 자료로 코호트부터 다시 생성
- WAR·PA/IP 등의 결과값은 계약 **후 3시즌**을 집계; 예측 특징량은 계약 당시 입수 가능한 정보로 제한

## 2. 실행 순서

```powershell
python -m pip install -r requirements.txt
python run_all.py
```

독립 검증(앞선 `run_all.py` 수행 후):

```powershell
python -m src.paper.validate_weight_grid_2class
python -m src.paper.validate_reproduction
```

공식 `run_all.py`는 이미 두 검증을 포함합니다. 단독으로 다시 실행할 때는 결과 파일이 필요합니다.

1. `src.modeling.run_temporal_final_models`
2. `src.evaluation.aav_external_validation`
3. `src.plot_final_model_contributions`
4. `src.visualization.figure1`
5. `src.paper.characterize_current_cutoffs`
6. `src.paper.build_paper_tables`
7. `src.paper.sensitivity_analysis_2and3class`
8. `src.paper.weight_grid_sensitivity_2class`
9. `src.paper.validate_weight_grid_2class`
10. `src.paper.make_paper_figures`
11. `src.paper.validate_reproduction`

## 3. 결과 계산 및 검증

**분류**

- 2-class `High`: Hitter Train Composite 상위 40%, SP/RP 상위 20%
- 3-class: Low/Middle/High 상대 성과집단
- 기본 Composite는 WAR와 PA/IP의 표준화 값에 같은 비중을 부여
- 정규화·라벨 컷오프는 학습 구간을 기준으로 정하고 Test 결과로 재설정하지 않음
- Accuracy, High-class F1, PR-AUC, Macro-F1 등을 결과와 함께 확인

**논문 비교**

- `outputs/paper/table4_three_class_composite.csv`: 3-class 대표 Composite 모델
- `outputs/paper/table5_binary_composite.csv`: 2-class 대표 Composite 모델
- `outputs/paper/table6_composite_vs_war_bootstrap.csv`: WAR-only 대비 Accuracy 및 paired bootstrap CI
- `outputs/paper/table7_aav_association.csv`: 계약 AAV와 사후 성과 관련성
- `outputs/paper/table8_sensitivity_recomputed.csv`: PCA와 2/3-class 라벨 일치율
- `outputs/paper/weight_grid/`: WAR 비중 0.0–1.0(간격 0.1) 총 33개 가중치·역할 조건의 예측/진단 결과

**Bootstrap**

paired percentile bootstrap은 같은 FA 계약의 Composite/WAR 결과를 짝지어 비교하며 10,000회 반복합니다. 재현 시 난수 생성 방식, 순서, 정렬 및 seed에 주의해야 합니다. AAV 상관계수 차이의 Bootstrap은 선수 단위(`player_key`)를 사용한 별도 분석입니다. 서로 다른 Bootstrap 결과를 같은 신뢰구간처럼 취급하지 마세요.

**최근 로컬 검증(2026-11-22)**

- `Reproduction validation: PASS` — **Passed 217, Failed 0**
- `All 33 conditions: metric/label checks PASS`
- `run_all.py` 종료: `Paper reproduction pipeline completed.`
- 보고서: `outputs/paper/REPRODUCTION_VALIDATION.md`

검증 PASS는 현재 코드에서 생성한 값이 검증 스크립트의 기대값과 일치한다는 의미입니다. 학습 데이터 구성, 모델 선정의 정당성, 논문 표기까지 자동으로 증명하는 것은 아닙니다.

## 4. 대표 모델 지정과 자동 선택의 차이 — 중요

`src/modeling/run_temporal_final_models.py`는 Logistic Regression, Random Forest, XGBoost 후보를 평가하고 `outputs/temporal_accuracy_final/temporal_best_models.csv`에 자동 선택 결과를 저장합니다.

그러나 논문 주요 표에 반영되는 모델은 `config/paper_model_spec.json`에 명시적으로 지정된 대표 모델이며, **자동 선택 결과와 일부 다릅니다.** 예컨대 이전 출력에서는 2-class Hitter/RP Composite의 자동 선택 결과가 XGBoost이고, 논문 대표 모델은 각각 Logistic Regression/Random Forest였습니다.

따라서 재현 실험은 다음을 분리하여 설명해야 합니다.

1. **후보모델 학습 및 평가**: 실제 후보를 학습하여 모든 지표 생성
2. **논문 표·그림 생성**: 버전 관리되는 대표 모델 스펙에 따라 해당 모델의 평가 결과 추출

이는 숨길 수 있는 단순 구현 세부사항이 아니라 **논문 방법절에 별도의 근거가 필요한 연구 설계 선택**입니다. `Validation Accuracy`만으로 모든 대표 모델이 선택됐다는 문장은 현재 코드와 일치하는지 확인하기 전에는 사용하지 마세요. 원고의 모델 선정 기준·후보군·보조 모델 위치를 일관되게 명시해야 합니다.

## 5. 민감도 분석과 성능 해석 유의점

- **절차 구분:** `sensitivity_analysis_2and3class.py`의 PCA 및 라벨 일치율은 전체 코호트를 사용하는 기술적 민감도 분석이며, Train-only 통계량으로 평가하는 예측 모델/weight-grid 실험과 전처리 범위가 다릅니다. 이 결과를 시간순 Test 예측성능의 근거처럼 취급하지 않습니다.
- PCA PC1 설명분산(재생성 결과, 소수 셋째 자리): Hitter **0.858**, SP **0.847**, RP **0.793**
- 인접 가중치 0.4:0.6 및 0.6:0.4에 대한 2/3-class 라벨 일치율을 재생성
- 확대 가중치 실험은 역할 3 × WAR 비중 11개 = **33조건**. 원고에서 제시할 0.1–0.9 중심 구간과 별도로 0.0 및 1.0은 경계 실험으로 다룸
- SP WAR 비중 0.0인 Validation 집단: **n=36, High=0**. 이 경우 High 분류 및 확률 임계값의 해석이 제한됨
- 기본 가중치 0.5의 Test Accuracy / High-F1: Hitter **0.783/0.618**, SP **0.723/0.308**, RP **0.898/0.000**
- 기본 가중치에서 SP/RP의 Accuracy가 Test 다수 클래스 기준보다 낮음. 이 점을 숨기지 말고 High-F1, PR-AUC 및 클래스 불균형과 함께 해석

이전 문서에 남아 있던 Table 8의 불일치 경고는 현재 재생성한 **표시 자릿수의 수치**와 별도로 검토해야 합니다. 2026-11-22 로컬 로그의 라벨 일치율은 기존 README에 나열됐던 값과 소수 셋째 자리에서 일치했지만, 원고 원본의 정밀도·산출 규칙·검증 코드 범위를 대조하지 않은 상태에서 모든 provenance 문제가 해결됐다고 단정하지 않습니다.

## 6. 원고 반영 전 확인 사항

- **모델 선택 근거:** 수동 스펙(`paper_model_spec.json`)과 자동 선택(`temporal_best_models.csv`) 간 차이 설명
- **Bootstrap CI:** 원고의 이전 타자 2-class 하한 **0.017**과 최신 재현 과정에서 보고된 **0.025** 차이를 최종 CSV 기준으로 확인하고 원고 표·본문에 일관되게 반영
- **중복 산출물:** 최종 RP 2-class는 RF인데 과거 `rp_2class_logistic_coefficients.csv`가 남아 있을 수 있음. 최종 RF 중요도(`rp_2class_rf_importance.csv`)와 혼동하지 않도록 결과 정리
- **Figure 9:** 과거 생성 스크립트는 공식 경로에서 분리됐으므로 현재 `run_all.py`가 해당 그림을 재생성한다고 안내하지 않음
- **재현 범위:** 기본 실행은 이미 만들어진 `processed_v2`를 읽음. 원시 데이터부터의 독립적 재현을 주장하려면 `--rebuild-cohort` 실행과 데이터 접근조건도 별도 확인

## 7. 저장소 정책

`src/`에는 공식 파이프라인용 코드를 남기고, 기존 최적화·Tertile·GMM·컷오프 확장 실험은 연구 이력 보존용 백업으로 분리합니다. 현재 Git 추적 대상에서 제외할 출력과 보존할 기준 스냅샷은 [`FILE_CLASSIFICATION.md`](FILE_CLASSIFICATION.md)를 참고하세요.

`outputs/reference/`는 이전 계산의 기준 스냅샷입니다. 공식 파이프라인이 생성하는 결과와 구분하고 임의로 덮어쓰거나 삭제하지 마세요.
