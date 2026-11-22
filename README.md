# MLB Free Agent Performance Prediction

## Paper Title

**English**  
Predicting the Performance of MLB Free Agents Upon Signing: A Machine Learning Approach

**한국어**  
MLB 자유계약선수의 계약 후 성과 예측: 머신러닝 기반 접근

---

## 1. Overview

This repository contains the code and data pipeline used for a machine-learning study on predicting the post-contract performance of Major League Baseball (MLB) free agents.

본 저장소는 MLB 자유계약선수(Free Agent, FA)의 계약 시점에 이용 가능한 정보를 활용하여 계약 후 성과를 예측한 연구의 재현용 코드와 데이터 처리 과정을 포함합니다.

본 연구에서는 FA 계약 이전의 선수 성과, 경기 참여량, 연령, 포지션 및 계약정보를 이용하여 계약 이후 3년간의 상대적 성과를 예측합니다.

선수의 역할에 따른 경기구조 차이를 고려하기 위해 분석대상을 다음과 같이 구분합니다.

- Hitter
- Starting Pitcher
- Relief Pitcher

각 역할에 대해 별도의 머신러닝 분류모델을 구축하고, 성과 정의 방식에 따른 예측결과의 차이를 비교합니다.

---

## 2. Research Objective

본 연구의 주요 목적은 다음과 같습니다.

1. FA 계약 당시 이용 가능한 정보만을 사용하여 계약 후 3년간의 성과를 예측
2. 기존의 WAR 중심 성과기준과 경기 참여 정도를 함께 고려한 Composite Score 비교
3. 타자, 선발투수, 불펜투수의 역할별 예측성능 비교
4. 2-class 및 3-class 분류를 통한 성과수준 예측
5. 계약 당시 시장평가를 나타내는 AAV와 성과지표 간의 관계 분석
6. Composite Score의 가중치 설정에 대한 민감도 검증

---

## 3. Data Sources

본 연구에서는 다음 두 공개 데이터 소스를 활용합니다.

### 3.1 Baseball-Reference

선수별 MLB 경기기록 및 성과지표를 수집합니다.

주요 변수 예시:

- WAR
- PA
- IP
- OPS+
- BB rate
- K rate
- ERA
- FIP
- K/BB
- GS
- G

### 3.2 Spotrac

FA 계약 관련 정보를 수집합니다.

주요 변수 예시:

- Contract year
- Contract length
- Contract value
- Average Annual Value (AAV)

---

## 4. Study Sample

분석대상은 계약 이후 3개 시즌의 성과를 관찰할 수 있는 MLB FA 계약으로 제한합니다.

| Role | Sample Size |
|---|---:|
| Hitter | 534 |
| Starting Pitcher | 216 |
| Relief Pitcher | 412 |
| **Total** | **1,162** |

분석기간:

- FA contract years: **2012–2021**

시간순 데이터 분할:

| Dataset | Contract Years |
|---|---|
| Train | 2012–2017 |
| Validation | 2018–2019 |
| Test | 2020–2021 |

시간순 데이터 분할을 사용함으로써 미래 계약정보가 과거 데이터의 학습과정에 포함되는 것을 방지합니다.

---

## 5. Performance Definition

### 5.1 WAR-only

계약 이후 3년간의 누적 WAR를 이용하여 선수의 상대적 성과를 정의합니다.

### 5.2 Composite Score

WAR만으로는 동일한 성과를 기록한 선수 간 실제 경기 참여량의 차이를 별도의 정보로 직접 구분하기 어렵다는 점을 고려하여, 본 연구에서는 계약 후 3년간의 WAR와 실제 경기 참여 정도를 결합한 Composite Score를 사용합니다.

역할별 활용도(Utilization) 변수는 다음과 같습니다.

- Hitter: PA
- Starting Pitcher: IP
- Relief Pitcher: IP

기본 Composite Score는 역할별로 표준화한 WAR와 활용도를 동일한 가중치로 결합합니다.

```text
Composite Score
= 0.5 × standardized WAR
+ 0.5 × standardized PA/IP
```

Composite Score는 WAR를 대체하는 지표라기보다, 누적 경기기여와 실제 경기 참여 정도를 함께 고려하기 위한 보완적 성과기준으로 사용합니다.

---

## 6. Classification Tasks

### 6.1 Binary Classification

2-class 분석에서는 상대적 고성과선수인 `High` 집단과 그 외 선수를 구분합니다.

역할별 High 기준:

- Hitter: top 40%
- Starting Pitcher: top 20%
- Relief Pitcher: top 20%

### 6.2 Three-Class Classification

3-class 분석에서는 역할별 성과분포의 1/3 및 2/3 분위수를 기준으로 다음 세 집단을 구성합니다.

- Low
- Middle
- High

따라서 본 연구의 성과집단은 계약의 절대적인 성공 또는 실패를 의미하지 않으며, 동일 역할의 FA 선수 내 상대적인 성과수준을 나타냅니다.

---

## 7. Machine Learning Models

본 연구에서는 다음 세 모델을 비교합니다.

- Logistic Regression
- Random Forest
- XGBoost

Logistic Regression은 해석 가능한 선형 기준모형으로 사용하며, Random Forest와 XGBoost는 비선형 관계와 변수 간 상호작용을 고려하기 위한 비교모형으로 사용합니다.

최종 논문 모델 선택에서는 Logistic Regression과 Random Forest를 주요 후보모형으로 사용하고, Validation Set의 Accuracy를 기준으로 최종 모델을 선택합니다.

Validation Accuracy가 동일한 경우에는 Balanced Accuracy를 보조 기준으로 사용합니다.

XGBoost는 추가적인 비교모형으로 학습 및 평가합니다.

---

## 8. Evaluation Metrics

### 8.1 Binary Classification

2-class 분석에서는 다음 지표를 확인합니다.

- Accuracy
- Precision
- Recall
- F1-score
- PR-AUC

클래스 불균형의 영향을 고려하기 위해 Accuracy뿐 아니라 Precision, Recall, F1-score 및 PR-AUC를 함께 확인합니다.

### 8.2 Three-Class Classification

3-class 분석에서는 다음 지표를 확인합니다.

- Accuracy
- Macro-F1
- Class-wise Precision
- Class-wise Recall
- Class-wise F1-score

---

## 9. Additional Analyses

### 9.1 Composite vs. WAR-only

동일한 입력변수, 시간분할 및 모델링 조건에서 결과라벨만 변경하여 Composite와 WAR-only의 예측성능을 비교합니다.

Composite와 WAR-only 간 Accuracy 차이의 불확실성을 평가하기 위해 paired percentile bootstrap을 수행합니다.

- Bootstrap repetitions: 10,000
- Resampling unit: player-contract observation
- Confidence interval: 95%

### 9.2 AAV Association Analysis

FA 계약 당시의 Average Annual Value(AAV)와 계약 이후 성과지표 간의 관계를 분석합니다.

비교 대상:

- AAV vs. WAR
- AAV vs. Composite Score

각 관계는 Spearman correlation을 이용하여 분석하며, WAR와 Composite Score의 AAV 관련성 차이도 함께 확인합니다.

### 9.3 Composite Weight Sensitivity

Composite Score의 기본 가중치인 `0.5:0.5`가 특정 가중치 설정에 지나치게 의존하는지를 확인하기 위해 다음과 같은 인접 가중치 조합을 추가로 비교합니다.

```text
WAR : Utilization

0.5 : 0.5   # Baseline
0.4 : 0.6
0.6 : 0.4
```

민감도 분석은 2-class와 3-class 모두에서 수행합니다.

#### 2-class

기본 가중치 `0.5:0.5`에서 생성된 High/Other 라벨과 대체 가중치에서 생성된 라벨의 일치율을 계산합니다.

#### 3-class

기본 가중치 `0.5:0.5`에서 생성된 Low/Middle/High 라벨과 대체 가중치에서 생성된 라벨의 일치율을 계산합니다.

또한 WAR와 활용도가 공통된 변동을 공유하는 정도를 확인하기 위해 PCA를 수행하고, 제1주성분(PC1)의 설명분산과 각 구성요소의 적재량을 확인합니다.

---

## 10. Repository Structure

```text
MLB_FA_Composite_Predict_Paper/
│
├── data/
│   ├── raw/
│   │   ├── batting/
│   │   └── pitching/
│   │
│   ├── processed/
│   │
│   └── processed_v2/
│       ├── hitter_cohort_v2.csv
│       ├── pitcher_cohort_v2.csv
│       ├── sp_cohort_v2.csv
│       └── rp_cohort_v2.csv
│
├── src/
│   ├── preprocessing/
│   │   └── build_analysis_cohort_v2.py
│   │
│   ├── modeling/
│   │   └── run_temporal_final_models.py
│   │
│   ├── evaluation/
│   │   └── aav_external_validation.py
│   │
│   ├── visualization/
│   │   └── figure1.py
│   │
│   └── paper/
│       ├── sensitivity_analysis.py
│       └── generate_figure9.py
│
├── outputs/
│   ├── temporal_accuracy_final/
│   ├── aav_external_validation/
│   ├── final_model_contributions/
│   └── paper/
│       ├── table8_sensitivity_recomputed.csv
│       └── figure9_construct_validity_weight_sensitivity.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 11. Reproduction

### Step 1. Clone the Repository

```bash
git clone <repository-url>
cd MLB_FA_Composite_Predict_Paper
```

`<repository-url>`은 실제 GitHub repository URL로 변경합니다.

### Step 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3. Install Dependencies

```bash
pip install -r requirements.txt
```

주요 Python packages:

- numpy
- pandas
- scipy
- scikit-learn
- xgboost
- matplotlib

---

## 12. Reproduce the Analysis

### 12.1 Build the Analysis Cohort

원시 Baseball-Reference 및 Spotrac 데이터를 이용하여 분석용 cohort를 다시 생성하려면 다음 명령을 실행합니다.

```bash
python src/preprocessing/build_analysis_cohort_v2.py
```

이미 `data/processed_v2/`에 저장된 최종 분석 cohort를 사용하는 경우 이 단계는 생략할 수 있습니다.

### 12.2 Run the Temporal Prediction Models

```bash
python src/modeling/run_temporal_final_models.py
```

이 단계에서는 역할별로 다음 분석을 수행합니다.

- WAR-only 2-class
- WAR-only 3-class
- Composite 2-class
- Composite 3-class

Train / Validation / Test 기간에 따른 모델 학습 및 평가를 수행하며, 주요 결과는 다음 위치에 저장됩니다.

```text
outputs/temporal_accuracy_final/
```

### 12.3 Run the AAV Association Analysis

```bash
python src/evaluation/aav_external_validation.py
```

주요 결과는 다음 위치에 저장됩니다.

```text
outputs/aav_external_validation/
```

### 12.4 Run the Composite Sensitivity Analysis

```bash
python src/paper/sensitivity_analysis.py
```

생성되는 주요 결과 파일:

```text
outputs/paper/table8_sensitivity_recomputed.csv
```

이 분석에서는 다음 값을 계산합니다.

- PC1 variance explained
- WAR loading
- Utilization loading
- 2-class label agreement
- 3-class label agreement

### 12.5 Generate Figure 9

Figure 9를 생성하기 전에 먼저 Composite sensitivity analysis를 실행해야 합니다.

```bash
python src/paper/generate_figure9.py
```

생성 파일:

```text
outputs/paper/figure9_construct_validity_weight_sensitivity.png
```

Figure 9는 다음 세 패널로 구성됩니다.

- (a) PC1 Loadings
- (b) 2-Class Label Agreement
- (c) 3-Class Label Agreement

---

## 13. Sensitivity Analysis Results

최종 분석 cohort에서 계산된 Composite Score 민감도 분석 결과는 다음과 같습니다.

| Role | PC1 Variance | WAR Loading | Utilization Loading | 2-class 0.4:0.6 | 2-class 0.6:0.4 | 3-class 0.4:0.6 | 3-class 0.6:0.4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hitter | 0.858 | 0.707 | 0.707 | 0.981 | 0.978 | 0.951 | 0.959 |
| Starting Pitcher | 0.847 | 0.707 | 0.707 | 0.972 | 0.991 | 0.963 | 0.954 |
| Relief Pitcher | 0.793 | 0.707 | 0.707 | 0.971 | 0.981 | 0.951 | 0.942 |

제1주성분의 설명분산은 Hitter 0.858, Starting Pitcher 0.847, Relief Pitcher 0.793으로 나타났으며, WAR와 Utilization의 적재량은 모든 역할에서 유사한 크기와 동일한 방향을 보였습니다.

2-class에서는 대체 가중치를 적용한 경우에도 기본 `0.5:0.5` 라벨과의 일치율이 모든 역할에서 0.971 이상으로 유지되었습니다.

3-class에서도 라벨 일치율은 0.942~0.963의 범위로 나타났으며, 2-class보다 일부 낮은 일치율을 보였습니다. 이는 Low, Middle, High 집단의 경계에 위치한 일부 선수들이 가중치 변화에 따라 다른 집단으로 이동할 수 있음을 보여줍니다.

전체적으로 Composite 기반 성과집단은 기본 `0.5:0.5`라는 단일 가중치 설정에만 의존하지 않고, 인접한 가중치 범위에서도 유사하게 유지되었습니다.

---

## 14. Reproducibility Notes

본 저장소는 논문 결과의 재현성을 높이기 위해 동일한 분석을 여러 코드에서 중복 계산하지 않는 것을 원칙으로 합니다.

예를 들어 Composite weight sensitivity 분석은 다음 구조로 실행됩니다.

```text
sensitivity_analysis.py
        ↓
table8_sensitivity_recomputed.csv
        ↓
generate_figure9.py
```

`generate_figure9.py`는 민감도 분석을 다시 계산하지 않고, `sensitivity_analysis.py`가 생성한 결과 파일을 읽어 Figure 9를 시각화합니다.

이를 통해 Table 8과 Figure 9가 서로 다른 계산방식을 사용하여 결과가 불일치하는 문제를 방지합니다.

---

## 15. Important Interpretation Notes

### 15.1 Composite Score

Composite Score는 WAR를 대체하는 새로운 절대적 선수평가 지표를 제안하기 위한 것이 아닙니다.

본 연구에서는 WAR가 나타내는 누적 경기기여와 실제 경기 참여 정도를 나타내는 PA 또는 IP를 함께 고려하여 FA 계약 이후 성과를 평가하는 보완적 성과기준으로 사용합니다.

PA와 IP는 선수의 순수한 능력만을 나타내는 변수가 아니라, 계약 후 실제로 얼마나 경기에 활용되었는지를 나타내는 realized utilization의 의미를 포함합니다.

따라서 활용도는 선수의 성과와 건강상태뿐 아니라 팀 상황, 보직, 로스터 구성 및 구단의 기용방식에도 영향을 받을 수 있습니다.

### 15.2 Relative Performance Groups

본 연구의 High / Middle / Low는 계약의 절대적인 성공 또는 실패를 의미하지 않습니다.

각 역할 내 FA 선수들의 계약 후 성과분포를 기준으로 정의된 상대적인 성과수준입니다.

따라서 High는 동일 역할의 다른 FA 선수들과 비교했을 때 상대적으로 높은 계약 후 성과를 기록한 집단을 의미합니다.

### 15.3 Prediction vs. Causality

본 연구는 예측을 목적으로 수행된 분석입니다.

모델에서 높은 변수 중요도 또는 계수를 보인 변수가 계약 후 성과의 인과적 원인임을 의미하지 않습니다.

모델의 변수 중요도와 계수는 주어진 입력변수 조건에서 분류과정에 활용된 예측정보로 해석해야 합니다.

---

## 16. Limitations

본 연구에는 다음과 같은 한계가 있습니다.

- 역할별 FA 표본 수가 제한적임
- 부상 이력 정보가 충분히 포함되지 않음
- Statcast 기반 구속 및 구종 정보가 포함되지 않음
- 구장효과와 팀 전력 변화가 충분히 반영되지 않음
- 계약 이후 보직변경 및 로스터 이동의 영향을 직접 모델링하지 못함
- PA와 IP는 선수의 능력뿐 아니라 건강상태, 팀 상황 및 구단의 기용방식에도 영향을 받음

따라서 예측결과는 개별 FA 선수의 성공 또는 실패를 확정적으로 판단하기 위한 결과가 아니라, 계약 당시 이용 가능한 정보를 기반으로 상대적인 계약 후 성과를 예측할 수 있는 가능성을 분석한 결과로 해석해야 합니다.

---

## 17. Future Work

향후 연구에서는 다음과 같은 확장이 가능합니다.

- Injury history
- Statcast-based features
- Pitch velocity and pitch mix
- Park factors
- Team strength
- Role transitions
- Direct regression of WAR / Composite Score
- Contract-value efficiency analysis
- Role-specific Composite weighting

---

## 18. Citation

If you use this repository, please cite the associated paper.

**English**

```text
Predicting the Performance of MLB Free Agents Upon Signing:
A Machine Learning Approach
```

**한국어**

```text
MLB 자유계약선수의 계약 후 성과 예측:
머신러닝 기반 접근
```

---

## 19. License

This repository is provided for academic and research purposes.

The data used in this study were collected from Baseball-Reference and Spotrac. Users should review the terms and conditions of the original data providers before redistributing raw data obtained from these sources.# MLB_FA_Composite_Prediction_Paper
