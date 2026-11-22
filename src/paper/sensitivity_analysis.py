from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'processed_v2'
OUT = ROOT / 'outputs' / 'paper'

CONFIG = {
    'hitter': ('Hitter', 'post_PA_3y', .40),
    'sp': ('Starting Pitcher', 'post_IP_3y', .20),
    'rp': ('Relief Pitcher', 'post_IP_3y', .20),
}


def top_label(score, rate):
    k = max(1, int(round(len(score) * rate)))
    return (score.rank(method='first', ascending=False) <= k).astype(int)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    weights = []
    for role, (label, playtime, top) in CONFIG.items():
        df = pd.read_csv(DATA/f'{role}_cohort_v2.csv')
        df = df[df.fa_year.between(2012, 2021)].copy()
        x = df[['post_WAR_3y_sum', playtime]].apply(pd.to_numeric, errors='coerce').dropna()
        z = (x - x.mean()) / x.std(ddof=0)
        pca = PCA(n_components=2).fit(z)
        load = pca.components_[0].copy()
        if load.sum() < 0:
            load *= -1
        equal = .5*z.iloc[:, 0] + .5*z.iloc[:, 1]
        base_label = top_label(equal, top)
        row = {
            'Role': label, 'n': len(z),
            'PC1 Variance Explained': pca.explained_variance_ratio_[0],
            'WAR Loading': load[0], 'Utilization Loading': load[1],
        }
        for war_w in [.4, .6]:
            score = war_w*z.iloc[:, 0] + (1-war_w)*z.iloc[:, 1]
            alt = top_label(score, top)
            agreement = float((base_label.to_numpy() == alt.to_numpy()).mean())
            row[f'Label Agreement ({war_w:.1f}:{1-war_w:.1f})'] = agreement
            weights.append({'Role': label, 'WAR Weight': war_w,
                            'Utilization Weight': 1-war_w, 'Label Agreement': agreement})
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'table8_sensitivity_recomputed.csv', index=False)
    pd.DataFrame(weights).to_csv(OUT/'weight_sensitivity_recomputed.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == '__main__':
    main()
