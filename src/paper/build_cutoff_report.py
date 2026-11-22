"""Generate Korean audit report and static scientific figures from cutoff outputs."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'outputs'/'cutoff_sensitivity_203040'
names={'hitter':'타자','sp':'선발','rp':'불펜'}
m=pd.read_csv(P/'paper_fixed_metrics.csv'); test=m[m.split.eq('test')&m.outcome.eq('composite')].copy()
c=pd.read_csv(P/'class_counts.csv'); prof=pd.read_csv(P/'outcome_profiles.csv')
b=pd.read_csv(P/'cluster_bootstrap_deltas.csv'); checks=pd.read_csv(P/'original_reproduction_checks.csv')
assert checks.labels_equal.all() and checks.predictions_equal.all()
assert len(pd.read_csv(P/'all_metrics.csv'))==108
assert len(test)==9
rows=[]
for r in test.itertuples():
 cc=c[c.role.eq(r.role)&c.top.eq(r.top)&c.outcome.eq('composite')].set_index('split')
 rows.append({'역할':names[r.role],'High 기준':f'{r.top:.0%}','훈련/검증/시험 High':f'{cc.loc["train","high"]}/{cc.loc["valid","high"]}/{cc.loc["test","high"]}','정확도':r.accuracy,'균형정확도':r.balanced_accuracy,'High 재현율':r.recall,'High F1':r.f1,'AP':r.pr_auc,'다수예측 대비 ΔAcc':r.accuracy_gain})
table=pd.DataFrame(rows)
profile=prof[prof.split.eq('train')&prof.outcome.eq('composite')&prof.high.eq(1)].copy()
profile['role']=profile.role.map(names);profile['top']=profile.top.map(lambda x:f'{x:.0%}')
profile=profile[['role','top','n','war_median','playtime_median','war_nonpositive_rate']]
profile.columns=['역할','High 기준','훈련 High 수','WAR 중앙값','PA/IP 중앙값','WAR≤0 비율']
cut=pd.read_csv(P/'label_parameters.csv')[['role','top','composite_2class_cut','war_2class_cut']]
cut['composite_2class_cut']=cut.composite_2class_cut/2
cut['role']=cut.role.map(names);cut['top']=cut.top.map(lambda x:f'{x:.0%}')
cut.columns=['역할','High 기준','0.5/0.5 Composite 경계','WAR-only 경계']
bt=b[b.policy.eq('paper_fixed')].copy();bt['role']=bt.role.map(names);bt['top']=bt.top.map(lambda x:f'{x:.0%}')
bt=bt[['role','top','delta_accuracy','bootstrap_se','ci_low','ci_high']]
bt.columns=['역할','High 기준','ΔAcc (Composite−WAR)','SE','95% CI 하한','95% CI 상한']
fig,axs=plt.subplots(1,3,figsize=(13,3.9),sharex=True)
for role,fr in test.groupby('role',sort=False):
 for ax,col,title in zip(axs,['balanced_accuracy','recall','f1'],['Balanced accuracy','High recall','High F1']):
  ax.plot(fr.top*100,fr[col],marker='o',label=role.upper(),linewidth=2)
  ax.set(title=title,xlabel='High group: top (%)',ylim=(-.03,1.03),xticks=[20,30,40]);ax.grid(alpha=.2)
axs[0].axhline(.5,color='gray',linestyle='--',linewidth=1);axs[-1].legend(frameon=False)
fig.suptitle('Cutoff sensitivity | paper-specified models | test 2020–2021',fontsize=13)
fig.tight_layout();fig.savefig(P/'cutoff_sensitivity.png',dpi=200);fig.savefig(P/'cutoff_sensitivity.pdf');plt.close(fig)
text='''# MLB FA 고성과 cutoff 민감도 분석 및 기준 선정 근거

## 핵심 판단

분석을 완료했지만, 현재 자료만으로 **타자 40%·선발/불펜 20%가 객관적으로 최적이라는 결론은 내릴 수 없다.** 20·30·40%는 고성과의 선별 범위를 단계적으로 바꾸는 시나리오로 정당화할 수 있다. 기존 역할별 설정은 연구자가 정한 조작적 정의로 제시하고, 이번 결과는 사후 민감도 분석으로 보고해야 한다.

타자 40%는 넓은 상위 성과군을 정의하고 훈련 High 131건을 확보한다는 설명이 가능하다. 선발·불펜 20%는 더 높은 기여·활용 수준의 상위 꼬리를 좁게 선별한다는 설명이 가능하다. 그러나 왜 타자에게는 넓은 정의, 투수에게는 좁은 정의가 필요한지는 별도의 연구 목적·실무 요구가 필요하다. 이번 분석으로 과거의 선정 동기를 입증하거나 만들어낼 수는 없다.

## 1. 실제 저장소에서 확인한 분석 구조

- 입력: `data/processed_v2/{hitter,sp,rp}_cohort_v2.csv`; 각각 534, 216, 412건. 계약 단위이므로 고유 선수 수와 다르다.
- 분할: 훈련 2012–2017, 검증 2018–2019, 시험 2020–2021. 모든 cutoff에서 동일 계약을 사용했다.
- 원본 `run_temporal_final_models.py`의 결과변수 변환, 특성, 전처리, LR/RF/XGB 하이퍼파라미터를 재사용했다.
- 표준화 평균·표준편차와 성과 cutoff는 훈련자료에서만 추정했다. High는 해당 경계값 이상(≥)이다.
- 본 분석: `config/paper_model_spec.json`의 모델을 고정했다. 타자·선발 LR, 불펜 RF이며 WAR-only에도 동일 역할 모델을 사용한다.
- 보조 분석: 각 cutoff와 결과변수에서 검증 정확도, 동률 시 균형정확도로 LR/RF/XGB 중 모델을 선택했다.
- 예측확률 임계값은 각 모델·cutoff의 검증자료에서 원본 함수로 결정했다. 이는 성과 라벨을 만드는 percentile cutoff와 다른 값이다.
- 기존 타자 40%, 선발·불펜 20%에서 Composite/WAR 총 6개 조합의 시험 라벨과 예측값이 모두 원본과 일치했다. 확률 차이는 부동소수점 수준(약 10^-16)이다.
- 원본의 `temporal_best_models.csv`와 논문 지정 모델은 일부 다르다. 본 보고서는 둘을 분리해 결과를 제공한다. 논문 지정 모델의 최초 선택 이유는 이 파일만으로 확인할 수 없다.
- `legacy_related/.../compare_2class_cutoffs.py`는 전체 자료 표준화·cutoff 산출 후 무작위 분할하는 구버전이다. 현재 시간 분할 실험의 선정 근거로 그대로 사용하지 않았다.

## 2. 20·30·40%라는 후보 범위를 사용하는 이유

| 후보 | 연구 설계상 의미 | 장점과 비용 |
|---|---|---|
| 상위 20% | 좁은 상위 성과군: 다섯 계약 중 약 한 계약을 High로 정의 | 높은 점수 집단에 집중하나 High 표본 감소·불균형 증가 |
| 상위 30% | 20%와 40% 사이의 중간 시나리오 | 10%p 간격에서 결론이 바뀌는지 확인 |
| 상위 40% | 넓은 상위 성과군: 전체의 절반 미만을 High로 유지 | High 학습 표본이 늘지만 고성과 집단의 평균적 수준이 낮아짐 |

이 후보 범위는 기존 연구 설정인 20%와 40%를 양끝으로 포함하고 중간 30%를 추가하는 **대칭적인 비교 설계**다. 보편적으로 합의된 야구 고성과 기준이라는 뜻은 아니다. 역할마다 같은 세 후보를 비교하여 특정 역할에 유리한 후보 범위만 사용하는 문제를 피했다. 10%·50%까지 포함한 전 범위 최적화는 수행하지 않았다.

## 3. 시험 성능: 논문 지정 모델 고정

AP는 원본 코드의 `average_precision_score`이며 원본 CSV의 PR-AUC 명칭에 대응한다. 양성 비율이 달라지므로 AP의 절대값만으로 cutoff 간 우열을 판단하면 안 된다. 추가 CSV에는 AP 기준선(시험 High 비율), AP/기준선 비율도 포함했다.

'''+table.to_markdown(index=False,floatfmt='.3f')+'''

다수예측 기준선은 훈련자료의 다수 클래스를 시험자료 전체에 예측한 정확도이다. 이번 모든 설정에서 다수 클래스는 Other였다. 학습한 모델의 High 탐지능력을 판정할 때에는 이 기준선과 균형정확도·재현율을 함께 봐야 한다.

### 타자

40%에서 High 재현율 0.538, F1 0.618, 균형정확도 0.720으로 세 후보 중 가장 높았다. 정확도는 20%가 0.833으로 높지만, Other만 예측해도 0.833이다. 40%는 0.783으로 다수예측 0.675보다 0.108 높다. 따라서 넓은 상위군의 식별과 표본 확보라는 관점에서는 40%의 실용적 해석을 뒷받침한다. 단, 이는 이미 관찰한 시험결과에 대한 사후 설명이며, 사전 최적화 근거가 아니다.

### 선발

20%에서 균형정확도 0.580, 재현율 0.364로 30%·40%보다 높았다. 그러나 정확도 0.723은 Other 전원 예측의 0.831보다 낮고, 검증 High가 5건, 시험 High가 11건뿐이다. 40%의 F1 0.327은 20%의 0.308보다 약간 높다. 따라서 20%가 모든 성능지표에서 최적이라고 설명하면 안 된다.

### 불펜

20%는 시험 High 8건 중 정답 0건이다. High로 예측한 1건은 오분류였으며, 정확도 0.898은 Other 전원 예측의 0.909보다 낮았다. 30%에서도 High 정답이 없고, 40%에서도 재현율은 0.071에 그친다. 따라서 **불펜 20%의 높은 정확도는 고성과 식별 성능의 근거가 아니다.** cutoff 변경만으로 해결되지 않는다는 점이 핵심이다. 추후 검증 단계에서 balanced accuracy 또는 F1 등 High 탐지 목적에 맞는 확률 임계값 선택을 별도 분석할 필요가 있다.

## 4. 훈련자료로 본 고성과 집단의 실질적 의미

'''+profile.to_markdown(index=False,floatfmt='.3f')+'''

PA/IP는 계약 후 3년 누적값의 집단 중앙값이다. 집단 중앙값은 개별 선수의 최소 자격조건이 아니다. 원본 IP 숫자 표현을 그대로 사용했으며 이 보고서에서 야구 이닝 소수 표기를 새로 변환하지 않았다.

- 타자 40%: High 131건, WAR 중앙값 3.30, PA 중앙값 1,295. 소수 최상위 계약에 한정하지 않고 기여와 활용을 함께 갖춘 넓은 상위군이라는 해석이 가능하다. 다만 WAR≤0 계약도 6.1% 포함된다.
- 선발 20%: High 23건, WAR 중앙값 8.70, IP 중앙값 535.00. 40%의 WAR 중앙값 4.80보다 높은 성과층을 선별한다.
- 불펜 20%: High 49건, WAR 중앙값 3.60, IP 중앙값 206.50. 40%의 WAR 중앙값 2.10·IP 중앙값 178.30보다 높은 성과층을 선별한다. 그렇더라도 미래 계약의 해당 집단을 잘 예측했다는 뜻은 아니다.

상위 비율을 줄일수록 점수가 높은 집단이 되는 것은 정의상 예상되는 현상이다. 이 표는 기준의 의미를 설명하지만 역할별 비율 차이를 독립적으로 입증하지는 않는다. WAR와 PA/IP는 Composite 자체의 구성요소이므로 이 표를 외적 타당도 검증으로 제시해서도 안 된다.

## 5. 실제 cutoff 값

'''+cut.to_markdown(index=False,floatfmt='.4f')+'''

표는 논문에서 사용 가능한 0.5·z(WAR)+0.5·z(PA/IP) 표기다. 모델 함수와 `label_parameters.csv`는 z(WAR)+z(PA/IP)를 사용하므로 CSV의 Composite 경계는 이 표의 2배다. 점수와 경계에 같은 양의 상수를 곱하므로 라벨은 동일하다. percentile은 훈련자료에서 정하므로 시험 High 비율이 명목상 20·30·40%와 같지 않다. 동점은 ≥ 규칙에 따라 모두 High로 포함한다.

## 6. Composite와 WAR-only의 차이는 cutoff에 강건한가?

동일 시험 계약을 짝지어 정확도 차이를 계산했다. 동일 선수의 여러 계약을 함께 재표집하는 선수 단위 cluster bootstrap 10,000회로 SE와 percentile 95% CI를 산출했다. 이는 원본의 계약 단위 bootstrap과 다르므로 CI가 정확히 같을 필요는 없다.

'''+bt.to_markdown(index=False,floatfmt='.4f')+'''

타자는 40%에서 ΔAcc=+0.133, 95% CI [0.023, 0.248]이지만 20%에서는 +0.008, 30%에서는 −0.033이다. 따라서 “Composite가 모든 고성과 기준에서 우수하다”는 결론은 지지되지 않는다. 기존 타자 결론은 40%라는 정의에 의존한다.

불펜 30%의 정확도 차이 CI가 0보다 높더라도 Composite High 재현율은 0이다. 두 결과변수의 양성 비율과 정답 라벨이 다르므로 이러한 차이가 Composite의 실제 고성과 식별 우수성을 입증하지 않는다. 선발에서는 세 cutoff 모두 차이의 CI가 0을 포함한다.

신뢰구간은 개별 비교에 대한 탐색적 구간이며 다중비교 보정은 하지 않았다. 모델 재학습·모델 선택·cutoff 선택의 불확실성을 모두 반영한 구간은 아니다.

## 7. 검증자료로 모델을 다시 선택한 보조 분석

타자 40%에서 XGB가 선택되어 정확도 0.750, High F1 0.605였다. WAR-only 대비 ΔAcc=+0.100이나 선수 cluster 95% CI는 약 [−0.016, 0.216]으로 0을 포함한다. 불펜 20%도 XGB가 선택되어 정확도 0.909, High 재현율 0이었다. 선발은 세 cutoff 모두 LR이 선택되었다. 본 분석의 질적 한계는 보조 분석에서도 해소되지 않았다. 전체 수치는 `validation_selected_metrics.csv`와 `cluster_bootstrap_deltas.csv`에 제공한다.

## 8. 논문에 사용할 수 있는 문장

### 방법: 후보 범위의 선정 근거

“이진 분류의 고성과 기준에 대한 결과의 민감도를 확인하기 위해 각 역할의 훈련자료에서 Composite Score 상위 20%, 30%, 40%에 해당하는 경계값을 각각 산출하였다. 상위 20%는 좁은 상위 성과군, 상위 40%는 보다 넓은 상위 성과군을 나타내며, 상위 30%는 두 정의 사이의 중간 시나리오로 설정하였다. 이 범위는 본 분석에서 사용한 비율을 포함하면서 고성과 집단의 선별 강도와 표본 수 변화에 따른 결과를 비교하기 위해 설정하였다. 각 경계값은 검증자료와 시험자료에 동일하게 적용하였다.”

### 기존 역할별 설정의 설명: 최적성 주장을 피하는 버전

“본 연구에서는 타자의 상대적으로 넓은 상위 성과군과 투수의 보다 좁은 상위 성과군을 구분하기 위해 타자 상위 40%, 선발 및 불펜투수 상위 20%를 조작적 고성과 기준으로 사용하였다. 훈련자료에서 해당 집단의 계약 후 3년 누적 WAR 중앙값은 각각 3.3, 8.7, 3.6이었으며, 활용도 중앙값은 타자 1,295 PA, 선발 535.0 IP, 불펜 206.5 IP였다. 다만 이러한 비율은 보편적인 고성과 기준이나 최적 경계값을 의미하지 않으며, 역할별 선별 강도가 다르다는 점을 고려하여 공통 비율을 적용한 민감도 분석을 추가하였다.”

위 문장은 사용한 정의를 설명하는 문장이지 과거의 의사결정 과정을 입증하는 문장이 아니다. 실제로 이러한 목적을 두고 정하지 않았다면 “~하기 위해”라는 표현을 제거하고 “타자에서는 ...를, 투수에서는 ...를 조작적 기준으로 사용하였다”라고만 기술한다. “역할별 특성을 반영하여 최적 기준을 결정하였다”는 문장은 현재 증거로 뒷받침되지 않는다.

### 결과 및 한계

“논문 지정 모델을 고정한 민감도 분석에서 타자 상위 40% 기준은 상위 20% 및 30% 기준보다 높은 High F1과 균형정확도를 보였다. 그러나 Composite와 WAR-only 간 정확도 차이는 상위 20%에서 0.008, 상위 30%에서 −0.033, 상위 40%에서 0.133으로 나타나, 비교 결과가 고성과 정의에 민감함을 확인하였다. 불펜투수의 상위 20% 기준에서는 높은 정확도에도 불구하고 High 재현율이 0이었으므로, 해당 결과를 고성과 선수 식별 능력의 근거로 해석하는 데 한계가 있다. 따라서 본 연구의 결과는 설정된 고성과 정의와 평가 절차의 범위에서 해석해야 한다.”

## 9. 저널 확장 시 결정해야 할 사항

1. 기존 비율을 유지한다면 조작적 정의로 명시하고 본 민감도 결과를 함께 보고한다. 성능이 좋았던 결과만 골라 보고하지 않는다.
2. 역할별 차등 기준이 반드시 필요하지 않다면 공통 기준을 본 분석으로 삼는 설계도 가능하다. 다만 지금 시험성능을 보고 공통 비율을 선택한 뒤 독립 검증처럼 보고해서는 안 된다.
3. 실제 의사결정 목적(예: 제한된 계약 후보군 선별, 일정 활용 수준 확보)을 먼저 정의하고, 훈련·검증 기간 내에서만 기준·평가 목적을 결정한다. 현재 분석에는 비용·효용 기반의 독립적 기준이 없다.
4. 불펜은 cutoff뿐 아니라 확률 임계값 선정 목적도 검토해야 한다. 검증 High가 8건인 설정에서 정확도 최적화는 소수 클래스 탐지와 잘 맞지 않을 수 있다.
5. 새 기간의 충분한 3년 성과 추적 자료 또는 훈련기간 내부 rolling validation으로 재확인한다. 현재 시험자료는 이미 논문과 이번 사후 분석에서 사용되었으므로 새로운 독립 검증자료가 아니다.
6. 동일 선수가 여러 기간에 등장한다. 선수 cluster bootstrap은 시험 내 반복 계약 의존성을 일부 반영하지만, 분할 간 반복 선수가 주는 모든 영향을 제거하지 않는다. `player_overlap.csv`에 현황을 제공한다.

## 10. 재현 방법 및 파일

Miniconda 예시(프로젝트 루트에서 실행):

```bash
conda create -n mlb-cutoff python=3.11 -y
conda activate mlb-cutoff
python -m pip install -r requirements.txt
python src/paper/cutoff_sensitivity_203040.py
python src/paper/build_cutoff_report.py
```

추가 스크립트는 기존 논문 표와 결과 파일을 변경하지 않고 `outputs/cutoff_sensitivity_203040/`에 기록한다. 3역할×3cutoff×2결과변수×3모델=54개 모델을 학습한다. 검증/시험 지표는 108행이다. 실제 실행 환경 버전과 원본 데이터 SHA-256은 `run_metadata.json`에 기록했다. 이번 런타임은 numpy/pandas가 저장소 고정 버전과 다르지만 기존 논문 설정의 라벨과 예측값은 모두 일치했다. 새로운 환경에서 재실행할 때에는 해당 재현 검사 파일도 확인한다.

- `paper_fixed_metrics.csv`: 논문 모델 고정 비교
- `validation_selected_metrics.csv`: 검증자료 모델 선택 보조 비교
- `all_metrics.csv`, `predictions.csv`: 모든 모델 지표 및 계약별 예측
- `class_counts.csv`, `labels.csv`, `outcome_profiles.csv`: 표본 수·라벨·집단 특성
- `label_parameters.csv`: 훈련자료 표준화 및 경계값
- `cluster_bootstrap_deltas.csv`: 두 정의 간 정확도 차이·SE·95% CI
- `original_reproduction_checks.csv`: 기존 결과 재현 확인
- `cutoff_sensitivity.png`, `.pdf`: 논문용 정적 성능 그래프

원본 압축파일에 포함된 가상환경과 캐시는 새 분석본 압축에서 제외했다. 원본 소스·데이터·기존 결과는 보존하고 분석 스크립트와 결과만 추가했다.
'''
(P/'cutoff_sensitivity_report_ko.md').write_text(text,encoding='utf-8')
table.to_csv(P/'summary_table_ko.csv',index=False,encoding='utf-8-sig')
print('Report and figures created; 6 original predictions reproduced; 108 metric rows verified.')
