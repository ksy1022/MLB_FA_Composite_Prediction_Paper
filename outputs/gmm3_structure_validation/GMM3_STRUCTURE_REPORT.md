# Three-component Composite structure validation

Only 2012–2017 training contracts were used for mixture fitting.
Validation/Test prediction performance was not used to choose the outcome groups.

## Main decision table

| Role | Best k | BIC margin | Modes | D L-M | D M-H | Weights L/M/H | High rate | Mean max posterior | Boot P(k=3 best) | Boot P(3 modes) | Clean 3-group support |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| Hitter | 3 | 11.2 | 2 | 2.35 | 2.02 | 0.42/0.42/0.17 | 13.1% | 0.87 | 77.0% | 1.3% | False |
| Starting Pitcher | 3 | 9.8 | 2 | 2.51 | 1.78 | 0.21/0.43/0.37 | 30.4% | 0.85 | 33.3% | 32.3% | False |
| Relief Pitcher | 3 | 6.2 | 3 | 2.25 | 5.12 | 0.41/0.55/0.04 | 3.7% | 0.92 | 81.0% | 51.3% | False |

## Interpretation rules

- `best_k_bic=3` alone is not enough.
- A genuine three-group interpretation is strongest when the 3-GMM density has three visible modes, adjacent components are well separated, assignments are confident, and the same structure reappears under player-cluster bootstrap.
- Component profiles should also be inspected. If Low/Middle/High do not show sensible WAR/Utilization patterns, do not describe them as performance levels merely because the Composite means are ordered.
- If `clean_3group_support=False` for a role, avoid using GMM boundaries as the main paper label for that role.

## Files to inspect

- `gmm3_combined_decision_summary.csv`
- `gmm3_component_profiles.csv`
- `gmm3_bootstrap_summary.csv`
- `{role}_gmm3_structure.png`
