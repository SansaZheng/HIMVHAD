"""Re-score18 frozen parents, independently audit scores/metrics; no NN fitting."""
import argparse,gc,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import genpareto
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,auc
from himvhad.model import HIMVHAnomalyDetector
from himvhad.data import load_data,load_views
from himvhad.preprocessing import Preprocessor
from himvhad.runner import rescore
from himvhad.audit import checkpoint_components,probe_rows
from himvhad.io import sha,write,provenance

SCORES=['field','flat','tail','tail_flat','skill_tail','skill_raw','skill_equal_tail','skill_hard_tail']

def close(a,b):
    if a is None or b is None:assert a is None and b is None
    else:np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)

def audit_metrics(folder,record,y):
    with np.load(folder/'scores.npz',allow_pickle=False) as z:c,s=z['calibration'],z['test']
    known=y!=-1;t=y[known];score=s[known];m=record['metrics'];p,r,_=precision_recall_curve(t,score)
    for k,v in dict(evaluated=len(t),unknown_excluded=int((~known).sum()),positives=int((t==1).sum()),normal=int((t==0).sum()),prevalence=float(t.mean()),average_precision=average_precision_score(t,score),roc_auc=roc_auc_score(t,score),pr_auc_trapezoid=auc(r,p)).items():close(m[k],v)
    th=json.loads((folder/'thresholds.json').read_text());u=np.quantile(c,.85);peaks=c[c>u]-u
    for name,v in th.items():
        if v['status']!='ok':assert len(peaks)<20 or np.ptp(peaks)==0;assert m['operating_points'][name] is None;continue
        if name=='quantile':threshold=np.quantile(c,.99)
        else:
            sh,_,sc=genpareto.fit(peaks,floc=0);close(v['shape'],sh);close(v['scale'],sc);threshold=u+genpareto.isf(.01*len(c)/len(peaks),sh,scale=sc)
        close(v['threshold'],threshold);a=score>=v['threshold'];tp=int(np.sum(a&(t==1)));fp=int(np.sum(a&(t==0)));fn=int(np.sum(~a&(t==1)));tn=int(np.sum(~a&(t==0)))
        vals=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),fpr=fp/(fp+tn),alarm_fraction=float(a.mean()))
        for k,value in vals.items():close(m['operating_points'][name][k],value)
    return c,s

def tail(c,x):
    result=[]
    for j in range(c.shape[1]):
        values,n=np.unique(c[:,j],return_counts=True);less=np.r_[0,np.cumsum(n)][np.searchsorted(values,x[:,j],side='left')];result.append(-np.log((len(c)+1-less)/(len(c)+1)))
    return np.column_stack(result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--old-parents',type=Path,required=True);p.add_argument('--new-results',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--parent-repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();torch.set_num_threads(4)
    old_published=json.loads((a.parent_repo/'results/roleweight_20261117/per_seed.json').read_text());parent_manifest=json.loads((a.new_results/'run_manifest.json').read_text());assert parent_manifest['git_commit']=='a683fcb152b8db48d5d6398f45e44331f605ac91'
    import hashlib
    for name,h in parent_manifest['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','-C',str(a.parent_repo),'show',parent_manifest['git_commit']+':'+name])).hexdigest()==h
    records=[];verified=[];weights=[];parents=[]
    # Fail on parsed fitting labels; byte hashes and data preparation are separate.
    original_fromfile=np.fromfile
    def label_guard(file,*args,**kw):
        if Path(file).name in ['train.y.i8','calibration.y.i8']:raise AssertionError('Fitting/calibration target access')
        return original_fromfile(file,*args,**kw)
    np.fromfile=label_guard
    for task in ['banksim','s_ffsd','cardsim']:
        schema,raw=load_data(a.data/task);view=load_views(a.data/task,'test')
        for seed in [42,43,44]:
            for base in ['m','cm']:
                group_start=time.monotonic();parent=(a.new_results if task=='cardsim' else a.old_parents)/'tabular'/task/'himvh'/base/str(seed);folder=a.output/'scores'/task/base/str(seed);model=HIMVHAnomalyDetector.load(parent/'model.pt','cpu')
                if task!='cardsim':
                    ref=next(r['parent_reference'] for r in old_published if (r['family'],r['dataset'],r['variant'],r['seed'])==('himvh',task,base+'_field',seed))
                    for name,h in ref['files_sha256'].items():assert sha(parent/name)==h
                else:
                    info=json.loads((parent/'parent_info.json').read_text());assert sha(parent/'model.pt')==info['checkpoint_sha256'] and sha(parent/'components.npz')==info['components_sha256']
                saved=torch.load(parent/'model.pt',map_location='cpu',weights_only=True)
                for name,v in saved['state'].items():assert torch.equal(v,model.net.state_dict()[name]),name
                assert model.width==len(schema['features']) and [g.tolist() for g in model.groups]==schema['field_groups'] and model.steps==800
                pre=Preprocessor.load(parent/'preprocessor.npz')
                with np.load(parent/'indices.npz',allow_pickle=False) as z:idx={k:z[k] for k in z.files}
                ti=np.sort(np.random.default_rng(seed).choice(len(raw['train']),min(8192,len(raw['train'])),replace=False));ci=np.sort(np.random.default_rng(seed+100).choice(len(raw['calibration']),min(100000,len(raw['calibration'])),replace=False))
                np.testing.assert_array_equal(ti,idx['train']);np.testing.assert_array_equal(ci,idx['calibration']);np.testing.assert_array_equal(idx['test'],np.arange(len(raw['test'])));fit=Preprocessor().fit(raw['train'][ti])
                xtr=pre.transform(raw['train'][ti]);xc=pre.transform(raw['calibration'][ci]);xt=pre.transform(raw['test'])
                for s,rows in [('train',ti),('calibration',ci),('test',np.arange(len(xt)))]:np.testing.assert_array_equal(pre.transform(raw[s][rows]),fit.transform(raw[s][rows]))
                with np.load(parent/'components.npz',allow_pickle=False) as z:ce,te=z['calibration'],z['test']
                rows=probe_rows(len(xt));probe=checkpoint_components(model,xt,view,rows);np.testing.assert_allclose(probe,te[rows],rtol=5e-4,atol=2e-5)
                r=rescore(parent,folder,SCORES,a.data/task)
                center=np.median(xtr.astype('f8'),axis=0);null=np.array([((xc[:,g].astype('f8')-center[g])**2).mean() for g in model.groups]);skill=np.clip(1-ce.mean(0)/np.maximum(null,1e-8),0,1);dims=np.array([len(g) for g in model.groups],float)
                def norm(w):return w/w.sum() if w.sum()>0 else dims/dims.sum()
                w=norm(dims*(.25+.75*skill));equal=norm(.25+.75*skill);hard=norm(dims*skill)
                with np.load(folder/'skill_calibrator.npz',allow_pickle=False) as z:
                    for key,value in dict(center=center,null_error=null,model_error=ce.mean(0),skill=skill,weights=w,equal=equal,hard=hard,widths=dims,sorted=np.sort(ce,axis=0)).items():close(z[key],value)
                values=[]
                for c in [ce,te]:
                    z=tail(ce,c);values.append(dict(field=c.mean(1),flat=np.average(c,axis=1,weights=dims),tail=z.mean(1),tail_flat=np.average(z,axis=1,weights=dims),skill_tail=z@w,skill_raw=c@w,skill_equal_tail=z@equal,skill_hard_tail=z@hard))
                y=np.fromfile(a.data/task/'test.y.i8',dtype='i1')
                for rec in r:
                    name=rec['scoring'];cal,test=audit_metrics(folder/name,rec,y);close(cal,values[0][name]);close(test,values[1][name]);gate=json.loads((folder/name/'label_access_gate.json').read_text());assert gate['parent_checkpoint_sha256']==sha(parent/'model.pt') and gate['parent_components_sha256']==sha(parent/'components.npz');assert gate['scores_sha256']==sha(folder/name/'scores.npz');assert gate['thresholds_sha256']==sha(folder/name/'thresholds.json');assert gate['field_calibrator_sha256']==sha(folder/'field_calibrator.npz') and gate['skill_calibrator_sha256']==sha(folder/'skill_calibrator.npz')
                    if task!='cardsim' and name in ['field','flat']:
                        oldv=base if name=='field' else base+'_flat'
                        with np.load(a.old_parents/'tabular'/task/'himvh'/oldv/str(seed)/'scores.npz',allow_pickle=False) as z:np.testing.assert_array_equal(cal,z['calibration']);np.testing.assert_array_equal(test,z['test'])
                    rec.update(dataset=task,variant=base,seed=seed,independent_synthetic_source=task=='cardsim',checkpoint_sha256=sha(parent/'model.pt'),components_sha256=sha(parent/'components.npz'),score_sha256=sha(folder/name/'scores.npz'),score_path=str((folder/name).relative_to(a.output)));records.append(rec)
                for j,g in enumerate(model.groups):weights.append(dict(dataset=task,variant=base,seed=seed,field=j,width=len(g),skill=float(skill[j]),weight=float(w[j]),null_error=float(null[j]),model_error=float(ce[:,j].mean())))
                parent_ref=dict(dataset=task,variant=base,seed=seed,training_source_commit=parent_manifest['git_commit'] if task=='cardsim' else '09de7f313d57470293f1309ed611914a88cbe62c',model_sha256=sha(parent/'model.pt'),files_sha256={n:sha(parent/n) for n in ['model.pt','preprocessor.npz','indices.npz','components.npz']});parents.append(parent_ref);verified.append(dict(dataset=task,variant=base,seed=seed,cpu_component_probe_max_abs=float(np.max(np.abs(probe-te[rows]))),checkpoint_state_exact=True,source_indices_exact=True));print('VERIFIED frozen group',len(parents),'/18',task,base,seed,'seconds',round(time.monotonic()-group_start,2),flush=True);del model,xt,xc,xtr;gc.collect()
    baselines=[]
    y=np.fromfile(a.data/'cardsim/test.y.i8',dtype='i1')
    for method in ['hbos','lof','vae']:
        for seed in [42,43,44]:
            root=a.new_results/'tabular/cardsim/baseline'/method/str(seed);rec=json.loads((root/'result.json').read_text());audit_metrics(root,rec,y)
            for name,h in rec['parent_sha256'].items():assert sha(root/name)==h
            baselines.append(rec)
    assert len(records)==144 and len(parents)==18 and len(baselines)==9
    write(a.output/'per_seed.json',records);write(a.output/'baseline_references.json',baselines);write(a.output/'weights.json',weights);write(a.output/'parent_models.json',parents);write(a.output/'verification.json',dict(status='passed',evaluated_configurations=144,operating_records=288,frozen_models=18,new_financial_network_fits=0,CPU_checkpoint_probes=verified,score_and_metric_replay=True,zero_fitting_labels_read=True,source=provenance(),parent_training_manifest=parent_manifest,new_validation_training_job='j-1117-8c114e',independent_source='single CardSim synthetic world; design frozen before generation,not real-bank external validation',baseline_reference_runs=9,seconds=time.monotonic()-started))
    print('Verified144 score configurations /288 operating points /18 frozen models /9 baseline references;no new financial NN fit')
if __name__=='__main__':main()
