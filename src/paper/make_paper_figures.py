from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs'/'paper'
TEMP = ROOT/'outputs'/'temporal_accuracy_final'


def annotate(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+.015, f'{h:.3f}', ha='center', va='bottom', fontsize=8)


def figure2():
    df = pd.read_csv(OUT/'table4_three_class_composite.csv')
    x = np.arange(len(df)); w=.34
    fig, ax = plt.subplots(figsize=(8.5,5))
    b1=ax.bar(x-w/2, df['Accuracy'], w, label='Accuracy')
    b2=ax.bar(x+w/2, df['Macro-F1'], w, label='Macro-F1')
    annotate(ax,b1); annotate(ax,b2)
    ax.set_xticks(x, df['Role']); ax.set_ylim(0,1); ax.set_ylabel('Score')
    ax.set_title('Selected Three-Class Composite Models under the Temporal Split')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT/'figure2_three_class_composite.png',dpi=300); plt.close(fig)


def figure3():
    df = pd.read_csv(OUT/'table5_binary_composite.csv')
    x = np.arange(len(df)); w=.34
    fig, ax = plt.subplots(figsize=(8.5,5))
    b1=ax.bar(x-w/2, df['Accuracy'], w, label='Accuracy')
    b2=ax.bar(x+w/2, df['F1'], w, label='High-class F1')
    annotate(ax,b1); annotate(ax,b2)
    ax.set_xticks(x, df['Role']); ax.set_ylim(0,1); ax.set_ylabel('Score')
    ax.set_title('Test Accuracy and High-Class F1 of Selected Binary Composite Models')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT/'figure3_binary_composite.png',dpi=300); plt.close(fig)


def figure4():
    pred = pd.read_csv(TEMP/'temporal_predictions.csv')
    spec = {'hitter':['logistic','rf'], 'sp':['logistic','rf'], 'rp':['logistic','rf']}
    labels={'hitter':'Hitter','sp':'Starting Pitcher','rp':'Relief Pitcher'}
    fig, axes = plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,role in zip(axes,['hitter','sp','rp']):
        sub0=pred[(pred.role==role)&(pred.outcome=='composite')&(pred.task=='2class')&(pred.split=='test')]
        baseline=sub0.y_true.mean()
        for model in spec[role]:
            sub=sub0[sub0.model==model]
            p, r, _ = precision_recall_curve(
                sub.y_true,
                sub.probability_1
            )

            ax.plot(
                r,
                p,
                label=f"{model.title()}"
            )
        ax.axhline(baseline,ls='--',lw=1,label=f'Baseline ({baseline:.3f})')
        ax.set_title(labels[role]); ax.set_xlabel('Recall'); ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    axes[0].set_ylabel('Precision'); axes[0].legend(fontsize=8)
    fig.suptitle('Precision-Recall Curves for Composite Binary Classification')
    fig.tight_layout(); fig.savefig(OUT/'figure4_precision_recall.png',dpi=300); plt.close(fig)


def comparison_figures():
    df=pd.read_csv(OUT/'table6_composite_vs_war_bootstrap.csv')
    for task, num in [('2class',7),('3class',8)]:
        d=df[df.Task==task]
        x=np.arange(len(d)); w=.34
        fig,ax=plt.subplots(figsize=(8.5,5))
        b1=ax.bar(x-w/2,d['Composite Accuracy'],w,label='Composite')
        b2=ax.bar(x+w/2,d['WAR-only Accuracy'],w,label='WAR-only')
        annotate(ax,b1);annotate(ax,b2)
        ax.set_xticks(x,d['Role']); ax.set_ylim(0,1); ax.set_ylabel('Accuracy');ax.legend()
        ax.set_title(f'Composite versus WAR-Only Accuracy ({task})')
        fig.tight_layout();fig.savefig(OUT/f'figure{num}_composite_vs_war_{task}.png',dpi=300);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    figure2(); figure3(); figure4(); comparison_figures()
    print('Saved figures to', OUT)

if __name__=='__main__': main()
