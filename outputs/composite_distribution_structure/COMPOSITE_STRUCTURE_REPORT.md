# Composite Score distribution-structure check

Only 2012–2017 training contracts were used to inspect the outcome distribution.
No validation/test prediction metric was used to define the candidate binary boundary.

## Results

| Role | ΔBIC 1−2 | Best k (BIC) | Modes in 2-GMM | Ashman D | Boundary | High rate | Bootstrap P(BIC2<1) | Bootstrap P(bimodal) | Clean 2-group support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hitter | 142.4 | 3 | 2 | 1.96 | -0.381 | 52.7% | 100.0% | 84.5% | False |
| Starting Pitcher | 6.9 | 3 | 1 | 2.19 | 0.287 | 31.3% | 94.5% | 51.5% | False |
| Relief Pitcher | 67.8 | 3 | 1 | 1.87 | 0.455 | 22.4% | 100.0% | 52.0% | False |

## Interpretation

- `best_k_bic = 3` means a 3-component Gaussian mixture fit has lower BIC than both 1- and 2-component fits.
- `two_component_mode_count = 1` means the fitted 2-component mixture is still visually unimodal; a second Gaussian component may only be modeling skewness or a tail.
- `Ashman D >= 2` is used here only as an empirical separation check.
- `clean_2group_support=False` means the script's conservative criteria do not support claiming a stable natural binary split.
