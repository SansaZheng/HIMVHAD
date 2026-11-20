"""Small CLI; full financial training is explicit, no implicit downloads."""
import argparse,json
from pathlib import Path
import numpy as np
from .runner import run_group,rescore,predict,SCORES
from .data import demo_data,prepare_csv
from .benchmark import fetch_benchmark
from .evaluation import metrics
from .thresholds import calibrate
from .io import write

def main():
    p=argparse.ArgumentParser(description='HIMVH zero-label adaptation (not supervised paper-score reproduction)');sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo',help='Tiny synthetic CPU smoke test');d.add_argument('--output',type=Path,default=Path('runs/demo'));d.add_argument('--steps',type=int,default=2)
    f=sub.add_parser('fetch-benchmark',help='Fetch pinned BankSim/S-FFSD caches and native keys');f.add_argument('--root',type=Path,default=Path('data/benchmark'));f.add_argument('--datasets',nargs='+',choices=['banksim','s_ffsd','cardsim'],default=['banksim','s_ffsd']);f.add_argument('--source-local',type=Path)
    c=sub.add_parser('prepare-csv');c.add_argument('--csv',type=Path,required=True);c.add_argument('--config',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--data',type=Path,required=True);r.add_argument('--output',type=Path,required=True);r.add_argument('--variants',nargs='+',choices=['legacy','f00','c','m','cm'],default=['cm']);r.add_argument('--seeds',nargs='+',type=int,default=[42]);r.add_argument('--steps',type=int);r.add_argument('--scores',nargs='+',choices=SCORES,default=['field','flat','tail','tail_flat']);r.add_argument('--device',default='cpu');r.add_argument('--hidden',type=int,default=256);r.add_argument('--max-train',type=int,default=8192);r.add_argument('--max-calibration',type=int,default=100000)
    s=sub.add_parser('rescore');s.add_argument('--parent',type=Path,required=True);s.add_argument('--output',type=Path,required=True);s.add_argument('--data',type=Path);s.add_argument('--scores',nargs='+',choices=SCORES,default=['field','flat','tail','tail_flat'])
    q=sub.add_parser('predict');q.add_argument('--run',type=Path,required=True);q.add_argument('--data',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--score',choices=SCORES,default='tail');q.add_argument('--phase',choices=['train','calibration','test'],default='test')
    e=sub.add_parser('evaluate');e.add_argument('--scores',type=Path,required=True);e.add_argument('--labels',type=Path,required=True);e.add_argument('--thresholds',type=Path);e.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.command=='demo':
        if a.output.exists():raise FileExistsError('Use a fresh demo folder')
        demo_data(a.output/'data');records=run_group(a.output/'data',a.output/'model',steps=a.steps,hidden=32,scores=['field','flat','tail','tail_flat']);write(a.output/'DEMO_ONLY.json',dict(financial_benchmark=False,description='Synthetic CLI smoke test; no financial performance claim'));print('DEMO complete (synthetic only): '+str(a.output))
    elif a.command=='fetch-benchmark':fetch_benchmark(a.root,a.datasets,a.source_local)
    elif a.command=='prepare-csv':prepare_csv(a.csv,a.config,a.output);print('Prepared chronological phases: '+str(a.output))
    elif a.command=='run':
        if a.output.exists():raise FileExistsError('Use a fresh benchmark output directory')
        if len(set(a.variants))!=len(a.variants) or len(set(a.seeds))!=len(a.seeds):raise ValueError('Duplicate variants/seeds')
        records=[]
        for seed in a.seeds:
            for variant in a.variants:
                records+=run_group(a.data,a.output/variant/str(seed),variant,seed,a.steps,a.scores,a.device,a.hidden,a.max_train,a.max_calibration);print('DONE '+variant+'/'+str(seed),flush=True)
        write(a.output/'records.json',records);write(a.output/'complete.json',dict(status='complete',network_fits=len(a.seeds)*len(a.variants),scoring_settings=len(records)))
    elif a.command=='rescore':rescore(a.parent,a.output,a.scores,a.data);print('Rescored frozen checkpoint without fitting a network')
    elif a.command=='predict':predict(a.run,a.data,a.output,a.score,a.phase);print('Predicted without target access: '+str(a.output))
    else:
        with np.load(a.scores,allow_pickle=False) as z:
            if 'test' in z and 'calibration' in z:c,t=z['calibration'],z['test']
            elif 'score' in z and a.thresholds is not None:c,t=None,z['score']
            else:raise ValueError('Use calibration/test scores, or predictions with --thresholds from the frozen run')
        y=np.load(a.labels,allow_pickle=False) if a.labels.suffix=='.npy' else np.fromfile(a.labels,dtype='i1');thresholds=json.loads(a.thresholds.read_text()) if a.thresholds else calibrate(c);result=metrics(y,t,thresholds)
        if a.output:write(a.output,result)
        else:print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
