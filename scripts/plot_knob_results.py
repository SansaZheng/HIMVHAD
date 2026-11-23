"""Static paired AP/F1 comparison for the scientific report."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    with (a.results/'summary.csv').open() as f:rows=list(csv.DictReader(f))
    variants=['baseline','cnn_fused','kan_fused','random_fused','uniform_fused','linear_fused','symbolic_fused']
    labels=['Original','CNN fusion','KAN fusion','Random knobs','Uniform layers','Linear rule','Symbolic rule']
    fig,axes=plt.subplots(2,3,figsize=(12,7.4),layout='constrained')
    colors=['#747b86','#4d9db4','#d47835','#9b8fc0','#7eaf82','#bc7d9a','#b6a05a']
    for col,dataset in enumerate(['banksim','s_ffsd','cardsim']):
        group=[next(r for r in rows if r['dataset']==dataset and r['variant']==v) for v in variants]
        for row,metric in enumerate(['average_precision','pot_f1']):
            ax=axes[row,col];mean=np.array([float(r[metric+'_mean'])*100 if r[metric+'_mean'] else np.nan for r in group]);sd=np.array([float(r[metric+'_sd'])*100 if r[metric+'_sd'] else 0 for r in group])
            ax.barh(np.arange(len(variants)),mean,xerr=sd,color=colors,error_kw=dict(elinewidth=.8,capsize=2))
            ax.set_yticks(np.arange(len(variants)),labels if col==0 else ['']*len(variants));ax.invert_yaxis()
            ax.set_xlabel('Average precision (%)' if row==0 else 'POT anomaly F1 (%)');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
            if row==0:ax.set_title({'banksim':'BankSim','s_ffsd':'S-FFSD','cardsim':'CardSim'}[dataset])
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen HIMVH + hidden-boundary adapter: three paired backbone seeds',fontsize=13)
    fig.savefig(a.results/'comparison.png',dpi=180);fig.savefig(a.results/'comparison.pdf');plt.close(fig)


if __name__=='__main__':main()
