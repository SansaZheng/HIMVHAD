"""Static scientific figure from audited CSV summaries."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results/frozen'));a=p.parse_args()
def read(path):
 with path.open() as f:rows=list(csv.DictReader(f))
 for r in rows:
  for k in r:
   if k.endswith(('_mean','_std')):r[k]=float(r[k]) if r[k] else None
 return rows
s=read(a.results/'summary.csv');b=read(a.results/'baseline_reference_summary.csv');ix={(r['dataset'],r['variant'],r['scoring'],r['operating_point']):r for r in s};fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
for col,d in enumerate(['banksim','s_ffsd','cardsim']):
 for row,m in enumerate(['average_precision','f1']):
  rs=[ix[d,'cm','field','pot'],ix[d,'cm','tail_flat','pot'],ix[d,'cm','skill_tail','pot'],ix[d,'m','skill_tail','pot']]+[next(x for x in b if (x['dataset'],x['method'],x['operating_point'])==(d,'hbos','pot'))];vals=[100*r[m+'_mean'] if r[m+'_mean'] is not None else np.nan for r in rs];sd=[100*r[m+'_std'] if r[m+'_std'] is not None else 0 for r in rs];ax=axes[row,col];ax.bar(np.arange(5),vals,yerr=sd,color=['#98b8cd','#5d99b8','#244f86','#6a7c9d','#999999'],capsize=3);ax.set_xticks(np.arange(5),['cm field','cm tail-flat','cm skill-tail','m skill-tail','HBOS ref'],rotation=30,ha='right');ax.set_ylabel(('AP' if row==0 else 'POT F1')+' (%)');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_title(d+(' (independent simulation)' if d=='cardsim' else ' (retrospective)'),fontsize=10)
fig.suptitle('Frozen HIMVH zero-label checkpoints | three seeds, mean ± sample SD | no new financial NN fitting',fontsize=11);fig.savefig(a.results/'comparison.png',dpi=180);fig.savefig(a.results/'comparison.pdf');plt.close(fig)
