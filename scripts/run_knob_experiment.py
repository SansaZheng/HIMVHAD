"""Pinned9-parent financial ablations; no real targets before every group freezes."""
import argparse
import gc
import json
import os
import resource
import shutil
import tarfile
import time
from pathlib import Path
import numpy as np
import torch
from himvhad.benchmark import fetch_benchmark
from himvhad.data import load_data,load_views
from himvhad.model import HIMVHAnomalyDetector
from himvhad.preprocessing import Preprocessor
from himvhad.knobs import KnobConfig,KnobAdapter,hidden_states,training_components
from himvhad.knob_runner import run_knobs,evaluate_knobs,VARIANTS
from himvhad.io import sha,write,provenance


def unpack(record, root, repo):
    src=repo/record['path'];assert sha(src)==record['sha256']
    root.mkdir(parents=True,exist_ok=False)
    with tarfile.open(src) as archive:
        members=archive.getmembers()
        if set(m.name for m in members)!=set(record['files_sha256']):raise ValueError('Unexpected parent archive members')
        for member in members:
            if not member.isfile() or '/' in member.name:raise ValueError('Unsafe parent archive member')
        archive.extractall(root,filter='data')
    for name,h in record['files_sha256'].items():assert sha(root/name)==h


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','full']);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--datasets',nargs='+',default=['banksim','s_ffsd','cardsim']);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);a=p.parse_args()
    root=a.output;root.mkdir(parents=True,exist_ok=True);repo=Path(__file__).resolve().parents[1];started=time.monotonic();torch.set_num_threads(4)
    if a.device.startswith('cuda'):
        assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.cuda.reset_peak_memory_stats()
    manifest=json.loads((repo/'experiments/knob_inputs/manifest.json').read_text());data=root/'scratch/data';fetch_benchmark(data,a.datasets)
    write(root/'runtime.json',dict(source=provenance(),stage=a.stage,device=a.device,gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None))
    original=np.fromfile
    def block_labels(file,*args,**kwargs):
        if str(file).endswith('.y.i8'):raise AssertionError('No label access during experiment construction')
        return original(file,*args,**kwargs)
    np.fromfile=block_labels
    if a.stage=='smoke':
        checks=[]
        for dataset in a.datasets:
            record=next(r for r in manifest['records'] if r['dataset']==dataset and r['seed']==42)
            parent=root/'scratch/parents'/dataset;unpack(record,parent,repo)
            schema,raw=load_data(data/dataset);model=HIMVHAnomalyDetector.load(parent/'model.pt',a.device);pre=Preprocessor.load(parent/'preprocessor.npz')
            # Actual largest phase context,8 label-independent target rows.
            x=pre.transform(raw['train']);views=load_views(data/dataset,'train');rows=np.linspace(0,len(x)-1,8,dtype='i8')
            states=hidden_states(model,x,views,rows);errors=training_components(model,x,views,rows,batch=8)
            assert np.isfinite(states).all() and np.isfinite(errors).all()
            checks.append(dict(dataset=dataset,context_rows=len(x),hidden_shape=list(states.shape),parent_sha256=sha(parent/'model.pt')))
            del model,x,views,states,errors;gc.collect()
        rng=np.random.default_rng(42);h=rng.normal(size=(256,5,256)).astype('f4');err=np.exp(h[:,0,0]);adapter=KnobAdapter(KnobConfig(probe_steps=2,student_steps=2,max_knobs=2),42,a.device).fit(h,err);adapter.save(root/'smoke_adapter.pt');assert np.isfinite(adapter.logits(h)['kan']).all()
        write(root/'smoke.json',dict(status='passed',checks=checks,financial_metrics=False,actual_labels_read=False,seconds=time.monotonic()-started,peak_ram_linux_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,peak_cuda_allocated_gb=torch.cuda.max_memory_allocated()/2**30 if a.device.startswith('cuda') else None))
    else:
        jobs=[r for r in manifest['records'] if r['dataset'] in a.datasets and r['seed'] in a.seeds];completed=[]
        for i,record in enumerate(jobs):
            dataset,seed=record['dataset'],record['seed'];parent=root/'scratch/parents'/dataset/str(seed);unpack(record,parent,repo)
            run=root/'runs'/dataset/str(seed);run_knobs(parent,data/dataset,run,KnobConfig(),a.device,evaluate=False)
            # Ensure paired baseline is exactly the historical published score.
            from himvhad.scoring import FieldScorer
            model=HIMVHAnomalyDetector.load(parent/'model.pt')
            with np.load(parent/'components.npz') as z:
                scorer=FieldScorer().fit(z['calibration'],[len(g) for g in model.groups])
                for phase in ['calibration','test']:
                    with np.load(run/'baseline/scores.npz') as saved:np.testing.assert_array_equal(saved[phase],scorer.all_scores(z[phase])['tail_flat'])
            completed.append((dataset,seed,run));del model;gc.collect()
            print(f'SCORED {i+1}/{len(jobs)} {dataset} seed{seed}',flush=True)
        write(root/'global_label_gate.json',dict(status='all_groups_frozen_before_any_test_label_read',groups=len(completed),gates={str(run.relative_to(root)):sha(run/'label_access_gate.json') for _,_,run in completed},source=provenance()))
        np.fromfile=original;records=[]
        for dataset,seed,run in completed:records+=evaluate_knobs(run,data/dataset)
        write(root/'records.json',records)
        write(root/'complete.json',dict(status='complete',groups=len(completed),configurations=len(records),new_backbone_fits=0,seconds=time.monotonic()-started,peak_ram_linux_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,peak_cuda_allocated_gb=torch.cuda.max_memory_allocated()/2**30 if a.device.startswith('cuda') else None))
    np.fromfile=original
    shutil.rmtree(root/'scratch')
    print('COMPLETE '+a.stage,flush=True)


if __name__=='__main__':main()
