from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/'paper'

EXPECTED6 = {
 ('Hitter','2class'):(.783,.650,.133,.017,.242),
 ('Starting Pitcher','2class'):(.723,.646,.077,-.015,.169),
 ('Relief Pitcher','2class'):(.898,.852,.046,-.011,.114),
 ('Hitter','3class'):(.483,.458,.025,-.092,.142),
 ('Starting Pitcher','3class'):(.385,.369,.016,-.138,.169),
 ('Relief Pitcher','3class'):(.420,.500,-.080,-.216,.057),
}

def main():
    d=pd.read_csv(OUT/'table6_composite_vs_war_bootstrap.csv')
    failures=[]
    for _,r in d.iterrows():
        exp=EXPECTED6[(r.Role,r.Task)]
        got=(r['Composite Accuracy'],r['WAR-only Accuracy'],r['Accuracy Difference'],r['CI Low 95%'],r['CI High 95%'])
        for g,e in zip(got,exp):
            if abs(float(g)-e)>1e-9:
                failures.append((r.Role,r.Task,got,exp));break
    status='PASS' if not failures else 'FAIL'
    text=[f'# Reproduction validation: {status}','', 'Table 6 values are checked against the manuscript at 3 decimals.']
    if failures:
        text += ['', 'Mismatches:', *[f'- {x}' for x in failures]]
    # Table 8 known provenance check
    if (OUT/'table8_sensitivity_recomputed.csv').exists():
        s=pd.read_csv(OUT/'table8_sensitivity_recomputed.csv')
        text += ['', '## Sensitivity-analysis provenance note',
                 'The current V2 cohorts reproduce the PCA values approximately, but the manuscript\'s exact Table 8 label-agreement values do not all match the surviving code/data. This is intentionally flagged rather than hard-coded.','', s.to_markdown(index=False)]
    (OUT/'REPRODUCTION_VALIDATION.md').write_text('\n'.join(text),encoding='utf-8')
    print('\n'.join(text[:5]))
    if failures: raise SystemExit(1)

if __name__=='__main__':main()
