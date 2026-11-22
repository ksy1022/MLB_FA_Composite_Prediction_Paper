# Reproduction validation: PASS

- Passed: 217
- Failed: 0
- Total checks: 217

## Validation results

| Check | Status | Detail |
|---|---|---|
| Cohort Hitter | PASS | actual=534.000000, expected=534.000000 |
| Hitter Train N | PASS | actual=328.000000, expected=328.000000 |
| Hitter Valid N | PASS | actual=86.000000, expected=86.000000 |
| Hitter Test N | PASS | actual=120.000000, expected=120.000000 |
| Hitter unique contract keys | PASS |  |
| Cohort Starting Pitcher | PASS | actual=216.000000, expected=216.000000 |
| Starting Pitcher Train N | PASS | actual=115.000000, expected=115.000000 |
| Starting Pitcher Valid N | PASS | actual=36.000000, expected=36.000000 |
| Starting Pitcher Test N | PASS | actual=65.000000, expected=65.000000 |
| Starting Pitcher unique contract keys | PASS |  |
| Cohort Relief Pitcher | PASS | actual=412.000000, expected=412.000000 |
| Relief Pitcher Train N | PASS | actual=241.000000, expected=241.000000 |
| Relief Pitcher Valid N | PASS | actual=83.000000, expected=83.000000 |
| Relief Pitcher Test N | PASS | actual=88.000000, expected=88.000000 |
| Relief Pitcher unique contract keys | PASS |  |
| Total cohort N | PASS | actual=1162.000000, expected=1162.000000 |
| Table 4 Hitter composite_cutoff | PASS | actual=-0.028388, expected=-0.028000 |
| Table 4 Hitter high_war_median | PASS | actual=3.300000, expected=3.300000 |
| Table 4 Hitter other_war_median | PASS | actual=-0.200000, expected=-0.200000 |
| Table 4 Hitter high_utilization_median | PASS | actual=1295.000000, expected=1295.000000 |
| Table 4 Hitter other_utilization_median | PASS | actual=327.000000, expected=327.000000 |
| Table 4 Starting Pitcher composite_cutoff | PASS | actual=0.801897, expected=0.802000 |
| Table 4 Starting Pitcher high_war_median | PASS | actual=8.700000, expected=8.700000 |
| Table 4 Starting Pitcher other_war_median | PASS | actual=0.300000, expected=0.300000 |
| Table 4 Starting Pitcher high_utilization_median | PASS | actual=535.000000, expected=535.000000 |
| Table 4 Starting Pitcher other_utilization_median | PASS | actual=194.500000, expected=194.500000 |
| Table 4 Relief Pitcher composite_cutoff | PASS | actual=0.602884, expected=0.603000 |
| Table 4 Relief Pitcher high_war_median | PASS | actual=3.600000, expected=3.600000 |
| Table 4 Relief Pitcher other_war_median | PASS | actual=0.000000, expected=0.000000 |
| Table 4 Relief Pitcher high_utilization_median | PASS | actual=206.500000, expected=206.500000 |
| Table 4 Relief Pitcher other_utilization_median | PASS | actual=69.750000, expected=69.750000 |
| Selected model count | PASS | actual=12.000000, expected=12.000000 |
| Validation model hitter/composite/2class | PASS | actual=logistic, expected=logistic |
| Saved model hitter/composite/2class | PASS |  |
| Validation model hitter/composite/3class | PASS | actual=rf, expected=rf |
| Saved model hitter/composite/3class | PASS |  |
| Validation model hitter/war/2class | PASS | actual=logistic, expected=logistic |
| Saved model hitter/war/2class | PASS |  |
| Validation model hitter/war/3class | PASS | actual=logistic, expected=logistic |
| Saved model hitter/war/3class | PASS |  |
| Validation model sp/composite/2class | PASS | actual=logistic, expected=logistic |
| Saved model sp/composite/2class | PASS |  |
| Validation model sp/composite/3class | PASS | actual=rf, expected=rf |
| Saved model sp/composite/3class | PASS |  |
| Validation model sp/war/2class | PASS | actual=logistic, expected=logistic |
| Saved model sp/war/2class | PASS |  |
| Validation model sp/war/3class | PASS | actual=rf, expected=rf |
| Saved model sp/war/3class | PASS |  |
| Validation model rp/composite/2class | PASS | actual=rf, expected=rf |
| Saved model rp/composite/2class | PASS |  |
| Validation model rp/composite/3class | PASS | actual=rf, expected=rf |
| Saved model rp/composite/3class | PASS |  |
| Validation model rp/war/2class | PASS | actual=rf, expected=rf |
| Saved model rp/war/2class | PASS |  |
| Validation model rp/war/3class | PASS | actual=logistic, expected=logistic |
| Saved model rp/war/3class | PASS |  |
| 3-class model Hitter | PASS |  |
| Table 5 Hitter Accuracy | PASS | actual=0.483333, expected=0.483000 |
| Table 5 Hitter Macro-F1 | PASS | actual=0.478710, expected=0.479000 |
| Table 5 Hitter High Recall | PASS | actual=0.696970, expected=0.697000 |
| Table 5 Hitter High F1 | PASS | actual=0.647887, expected=0.648000 |
| 2-class model Hitter | PASS |  |
| Table 6 Hitter Threshold | PASS | actual=0.700000, expected=0.700000 |
| Table 6 Hitter Accuracy | PASS | actual=0.783333, expected=0.783000 |
| Table 6 Hitter Precision | PASS | actual=0.724138, expected=0.724000 |
| Table 6 Hitter Recall | PASS | actual=0.538462, expected=0.538000 |
| Table 6 Hitter F1 | PASS | actual=0.617647, expected=0.618000 |
| Table 6 Hitter PR-AUC | PASS | actual=0.716035, expected=0.716000 |
| Prediction N Hitter/2class | PASS | actual=120.000000, expected=120.000000 |
| Recomputed Accuracy Hitter/2class | PASS | actual=0.783333, expected=0.783000 |
| Recomputed F1 Hitter | PASS | actual=0.617647, expected=0.618000 |
| Recomputed AP Hitter | PASS | actual=0.716035, expected=0.716000 |
| Prediction N Hitter/3class | PASS | actual=120.000000, expected=120.000000 |
| Recomputed Accuracy Hitter/3class | PASS | actual=0.483333, expected=0.483000 |
| Recomputed Macro-F1 Hitter | PASS | actual=0.478710, expected=0.479000 |
| Recomputed High Recall Hitter | PASS | actual=0.696970, expected=0.697000 |
| Recomputed High F1 Hitter | PASS | actual=0.647887, expected=0.648000 |
| 3-class model Starting Pitcher | PASS |  |
| Table 5 Starting Pitcher Accuracy | PASS | actual=0.384615, expected=0.385000 |
| Table 5 Starting Pitcher Macro-F1 | PASS | actual=0.363996, expected=0.364000 |
| Table 5 Starting Pitcher High Recall | PASS | actual=0.333333, expected=0.333000 |
| Table 5 Starting Pitcher High F1 | PASS | actual=0.341463, expected=0.341000 |
| 2-class model Starting Pitcher | PASS |  |
| Table 6 Starting Pitcher Threshold | PASS | actual=0.710000, expected=0.710000 |
| Table 6 Starting Pitcher Accuracy | PASS | actual=0.723077, expected=0.723000 |
| Table 6 Starting Pitcher Precision | PASS | actual=0.266667, expected=0.267000 |
| Table 6 Starting Pitcher Recall | PASS | actual=0.363636, expected=0.364000 |
| Table 6 Starting Pitcher F1 | PASS | actual=0.307692, expected=0.308000 |
| Table 6 Starting Pitcher PR-AUC | PASS | actual=0.355513, expected=0.356000 |
| Prediction N Starting Pitcher/2class | PASS | actual=65.000000, expected=65.000000 |
| Recomputed Accuracy Starting Pitcher/2class | PASS | actual=0.723077, expected=0.723000 |
| Recomputed F1 Starting Pitcher | PASS | actual=0.307692, expected=0.308000 |
| Recomputed AP Starting Pitcher | PASS | actual=0.355513, expected=0.356000 |
| Prediction N Starting Pitcher/3class | PASS | actual=65.000000, expected=65.000000 |
| Recomputed Accuracy Starting Pitcher/3class | PASS | actual=0.384615, expected=0.385000 |
| Recomputed Macro-F1 Starting Pitcher | PASS | actual=0.363996, expected=0.364000 |
| Recomputed High Recall Starting Pitcher | PASS | actual=0.333333, expected=0.333000 |
| Recomputed High F1 Starting Pitcher | PASS | actual=0.341463, expected=0.341000 |
| 3-class model Relief Pitcher | PASS |  |
| Table 5 Relief Pitcher Accuracy | PASS | actual=0.420455, expected=0.420000 |
| Table 5 Relief Pitcher Macro-F1 | PASS | actual=0.397854, expected=0.398000 |
| Table 5 Relief Pitcher High Recall | PASS | actual=0.238095, expected=0.238000 |
| Table 5 Relief Pitcher High F1 | PASS | actual=0.285714, expected=0.286000 |
| 2-class model Relief Pitcher | PASS |  |
| Table 6 Relief Pitcher Threshold | PASS | actual=0.640000, expected=0.640000 |
| Table 6 Relief Pitcher Accuracy | PASS | actual=0.897727, expected=0.898000 |
| Table 6 Relief Pitcher Precision | PASS | actual=0.000000, expected=0.000000 |
| Table 6 Relief Pitcher Recall | PASS | actual=0.000000, expected=0.000000 |
| Table 6 Relief Pitcher F1 | PASS | actual=0.000000, expected=0.000000 |
| Table 6 Relief Pitcher PR-AUC | PASS | actual=0.175756, expected=0.176000 |
| Prediction N Relief Pitcher/2class | PASS | actual=88.000000, expected=88.000000 |
| Recomputed Accuracy Relief Pitcher/2class | PASS | actual=0.897727, expected=0.898000 |
| Recomputed F1 Relief Pitcher | PASS | actual=0.000000, expected=0.000000 |
| Recomputed AP Relief Pitcher | PASS | actual=0.175756, expected=0.176000 |
| Prediction N Relief Pitcher/3class | PASS | actual=88.000000, expected=88.000000 |
| Recomputed Accuracy Relief Pitcher/3class | PASS | actual=0.420455, expected=0.420000 |
| Recomputed Macro-F1 Relief Pitcher | PASS | actual=0.397854, expected=0.398000 |
| Recomputed High Recall Relief Pitcher | PASS | actual=0.238095, expected=0.238000 |
| Recomputed High F1 Relief Pitcher | PASS | actual=0.285714, expected=0.286000 |
| Bootstrap comparison rows | PASS | actual=6.000000, expected=6.000000 |
| Table 7 Hitter/2class Composite Accuracy | PASS | actual=0.783000, expected=0.783000 |
| Table 7 Hitter/2class WAR-only Accuracy | PASS | actual=0.650000, expected=0.650000 |
| Table 7 Hitter/2class Accuracy Difference | PASS | actual=0.133000, expected=0.133000 |
| Table 7 Hitter/2class Bootstrap Mean Difference | PASS | actual=0.134000, expected=0.134000 |
| Table 7 Hitter/2class Bootstrap SE | PASS | actual=0.056000, expected=0.056000 |
| Table 7 Hitter/2class CI Low 95% | PASS | actual=0.025000, expected=0.025000 |
| Table 7 Hitter/2class CI High 95% | PASS | actual=0.242000, expected=0.242000 |
| Table 7 Hitter/3class Composite Accuracy | PASS | actual=0.483000, expected=0.483000 |
| Table 7 Hitter/3class WAR-only Accuracy | PASS | actual=0.458000, expected=0.458000 |
| Table 7 Hitter/3class Accuracy Difference | PASS | actual=0.025000, expected=0.025000 |
| Table 7 Hitter/3class Bootstrap Mean Difference | PASS | actual=0.026000, expected=0.026000 |
| Table 7 Hitter/3class Bootstrap SE | PASS | actual=0.060000, expected=0.060000 |
| Table 7 Hitter/3class CI Low 95% | PASS | actual=-0.092000, expected=-0.092000 |
| Table 7 Hitter/3class CI High 95% | PASS | actual=0.142000, expected=0.142000 |
| Table 7 Starting Pitcher/2class Composite Accuracy | PASS | actual=0.723000, expected=0.723000 |
| Table 7 Starting Pitcher/2class WAR-only Accuracy | PASS | actual=0.646000, expected=0.646000 |
| Table 7 Starting Pitcher/2class Accuracy Difference | PASS | actual=0.077000, expected=0.077000 |
| Table 7 Starting Pitcher/2class Bootstrap Mean Difference | PASS | actual=0.077000, expected=0.077000 |
| Table 7 Starting Pitcher/2class Bootstrap SE | PASS | actual=0.045000, expected=0.045000 |
| Table 7 Starting Pitcher/2class CI Low 95% | PASS | actual=-0.015000, expected=-0.015000 |
| Table 7 Starting Pitcher/2class CI High 95% | PASS | actual=0.169000, expected=0.169000 |
| Table 7 Starting Pitcher/3class Composite Accuracy | PASS | actual=0.385000, expected=0.385000 |
| Table 7 Starting Pitcher/3class WAR-only Accuracy | PASS | actual=0.369000, expected=0.369000 |
| Table 7 Starting Pitcher/3class Accuracy Difference | PASS | actual=0.016000, expected=0.016000 |
| Table 7 Starting Pitcher/3class Bootstrap Mean Difference | PASS | actual=0.014000, expected=0.014000 |
| Table 7 Starting Pitcher/3class Bootstrap SE | PASS | actual=0.077000, expected=0.077000 |
| Table 7 Starting Pitcher/3class CI Low 95% | PASS | actual=-0.138000, expected=-0.138000 |
| Table 7 Starting Pitcher/3class CI High 95% | PASS | actual=0.169000, expected=0.169000 |
| Table 7 Relief Pitcher/2class Composite Accuracy | PASS | actual=0.898000, expected=0.898000 |
| Table 7 Relief Pitcher/2class WAR-only Accuracy | PASS | actual=0.852000, expected=0.852000 |
| Table 7 Relief Pitcher/2class Accuracy Difference | PASS | actual=0.046000, expected=0.046000 |
| Table 7 Relief Pitcher/2class Bootstrap Mean Difference | PASS | actual=0.046000, expected=0.046000 |
| Table 7 Relief Pitcher/2class Bootstrap SE | PASS | actual=0.032000, expected=0.032000 |
| Table 7 Relief Pitcher/2class CI Low 95% | PASS | actual=-0.011000, expected=-0.011000 |
| Table 7 Relief Pitcher/2class CI High 95% | PASS | actual=0.114000, expected=0.114000 |
| Table 7 Relief Pitcher/3class Composite Accuracy | PASS | actual=0.420000, expected=0.420000 |
| Table 7 Relief Pitcher/3class WAR-only Accuracy | PASS | actual=0.500000, expected=0.500000 |
| Table 7 Relief Pitcher/3class Accuracy Difference | PASS | actual=-0.080000, expected=-0.080000 |
| Table 7 Relief Pitcher/3class Bootstrap Mean Difference | PASS | actual=-0.079000, expected=-0.079000 |
| Table 7 Relief Pitcher/3class Bootstrap SE | PASS | actual=0.071000, expected=0.071000 |
| Table 7 Relief Pitcher/3class CI Low 95% | PASS | actual=-0.216000, expected=-0.216000 |
| Table 7 Relief Pitcher/3class CI High 95% | PASS | actual=0.057000, expected=0.057000 |
| Table 8 Hitter AAV-WAR rho | PASS | actual=0.395000, expected=0.395000 |
| Table 8 Hitter AAV-Composite rho | PASS | actual=0.587000, expected=0.587000 |
| Table 8 Hitter Difference | PASS | actual=0.192000, expected=0.192000 |
| Table 8 Hitter CI Low 95% | PASS | actual=0.143000, expected=0.143000 |
| Table 8 Hitter CI High 95% | PASS | actual=0.244000, expected=0.244000 |
| AAV Bootstrap repetitions Hitter | PASS | actual=10000.000000, expected=10000.000000 |
| AAV Bootstrap unit Hitter | PASS |  |
| Table 8 Starting Pitcher AAV-WAR rho | PASS | actual=0.372000, expected=0.372000 |
| Table 8 Starting Pitcher AAV-Composite rho | PASS | actual=0.467000, expected=0.467000 |
| Table 8 Starting Pitcher Difference | PASS | actual=0.095000, expected=0.095000 |
| Table 8 Starting Pitcher CI Low 95% | PASS | actual=0.037000, expected=0.037000 |
| Table 8 Starting Pitcher CI High 95% | PASS | actual=0.157000, expected=0.157000 |
| AAV Bootstrap repetitions Starting Pitcher | PASS | actual=10000.000000, expected=10000.000000 |
| AAV Bootstrap unit Starting Pitcher | PASS |  |
| Table 8 Relief Pitcher AAV-WAR rho | PASS | actual=0.209000, expected=0.209000 |
| Table 8 Relief Pitcher AAV-Composite rho | PASS | actual=0.336000, expected=0.336000 |
| Table 8 Relief Pitcher Difference | PASS | actual=0.127000, expected=0.127000 |
| Table 8 Relief Pitcher CI Low 95% | PASS | actual=0.078000, expected=0.078000 |
| Table 8 Relief Pitcher CI High 95% | PASS | actual=0.180000, expected=0.180000 |
| AAV Bootstrap repetitions Relief Pitcher | PASS | actual=10000.000000, expected=10000.000000 |
| AAV Bootstrap unit Relief Pitcher | PASS |  |
| PCA Hitter | PASS | actual=0.858036, expected=0.858000 |
| PCA Starting Pitcher | PASS | actual=0.846818, expected=0.847000 |
| PCA Relief Pitcher | PASS | actual=0.793372, expected=0.793000 |
| 2-class agreement columns | PASS |  |
| 3-class agreement columns | PASS |  |
| 2-class agreement minimum | PASS | actual=0.970874, expected=0.971000 |
| 2-class agreement maximum | PASS | actual=0.990741, expected=0.991000 |
| 3-class agreement minimum | PASS | actual=0.941748, expected=0.942000 |
| 3-class agreement maximum | PASS | actual=0.962963, expected=0.963000 |
| Weight-grid conditions | PASS | actual=33.000000, expected=33.000000 |
| Grid rows hitter | PASS | actual=11.000000, expected=11.000000 |
| Grid baseline N hitter | PASS | actual=120.000000, expected=120.000000 |
| Grid baseline Accuracy hitter | PASS | actual=0.783333, expected=0.783000 |
| Grid baseline High F1 hitter | PASS | actual=0.617647, expected=0.618000 |
| Grid rows sp | PASS | actual=11.000000, expected=11.000000 |
| Grid baseline N sp | PASS | actual=65.000000, expected=65.000000 |
| Grid baseline Accuracy sp | PASS | actual=0.723077, expected=0.723000 |
| Grid baseline High F1 sp | PASS | actual=0.307692, expected=0.308000 |
| Grid rows rp | PASS | actual=11.000000, expected=11.000000 |
| Grid baseline N rp | PASS | actual=88.000000, expected=88.000000 |
| Grid baseline Accuracy rp | PASS | actual=0.897727, expected=0.898000 |
| Grid baseline High F1 rp | PASS | actual=0.000000, expected=0.000000 |
| Contribution model hitter/2class | PASS | expected=logistic |
| Contribution model hitter/3class | PASS | expected=rf |
| RF importance sum hitter/3class | PASS | actual=1.000000, expected=1.000000 |
| Contribution model sp/2class | PASS | expected=logistic |
| Contribution model sp/3class | PASS | expected=rf |
| RF importance sum sp/3class | PASS | actual=1.000000, expected=1.000000 |
| Contribution model rp/2class | PASS | expected=rf |
| RF importance sum rp/2class | PASS | actual=1.000000, expected=1.000000 |
| Contribution model rp/3class | PASS | expected=rf |
| RF importance sum rp/3class | PASS | actual=1.000000, expected=1.000000 |
| Figure exists: figure5_hitter_model_contributions.png | PASS |  |
| Figure exists: figure6_pitcher_model_contributions.png | PASS |  |