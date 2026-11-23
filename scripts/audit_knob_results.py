"""Independent paired-score/threshold/metric replay after CQ collection."""
import argparse
import json
import tarfile
import io
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,auc
from himvhad.io import sha,write
from himvhad.knob_runner import VARIANTS


def assertclose(a,b):
    if a is None or b is None:assert a is None and b is None
    elif isinstance(a,(dict,list)):assert a==b
    else:np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--code-commit',required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];manifest=json.loads((repo/'experiments/knob_inputs/manifest.json').read_text());root=a.input
    records=json.loads((root/'records.json').read_text());globalgate=json.loads((root/'global_label_gate.json').read_text());assert len(records)==81 and globalgate['groups']==9
    historic=json.loads((repo/'results/frozen/per_seed.json').read_text());checks=[]
    runtime=json.loads((root/'runtime.json').read_text())
    commit=a.code_commit
    for name,h in runtime['source']['source_sha256'].items():
        content=subprocess.check_output(['git','-C',str(repo),'show',commit+':himvhad/'+name])
        assert hashlib.sha256(content).hexdigest()==h,'Runtime source differs from pinned commit: '+name
    for item in manifest['records']:
        dataset,seed=item['dataset'],item['seed'];folder=root/'runs'/dataset/str(seed);gate=json.loads((folder/'label_access_gate.json').read_text())
        info=json.loads((folder/'adapter_info.json').read_text())
        if not a.pilot:assert info['selected_student_and_controls_rng_seed']==seed+1000+info['selected_k'],'Unpaired pilot must not be used as final ablation'
        assert sha(folder/'label_access_gate.json')==globalgate['gates'][str(folder.relative_to(root))]
        for name,h in gate['files_sha256'].items():assert sha(folder/name)==h,name
        with tarfile.open(repo/item['path']) as t:
            blob=t.extractfile('components.npz').read()
        with np.load(io.BytesIO(blob),allow_pickle=False) as z:components={s:z[s] for s in ['calibration','test']}
        schema=json.loads((a.data/dataset/'schema.json').read_text());widths=np.array([len(g) for g in schema['field_groups']])
        baseline={}
        for phase in ['calibration','test']:
            ce,xe=components['calibration'],components[phase]
            evidence=np.column_stack([-np.log((len(ce)+1-np.searchsorted(np.sort(ce[:,g]),xe[:,g],side='left'))/(len(ce)+1)) for g in range(ce.shape[1])])
            baseline[phase]=np.average(evidence,axis=1,weights=widths)
        y=np.fromfile(a.data/dataset/'test.y.i8',dtype='i1');known=y!=-1;target=y[known]
        with np.load(folder/'baseline/scores.npz') as z:saved_baseline={s:z[s] for s in ['calibration','test']}
        baseline_delta={s:float(np.max(np.abs(saved_baseline[s]-baseline[s]))) for s in baseline}
        baseline_exact=all(np.array_equal(saved_baseline[s],baseline[s]) for s in baseline)
        with np.load(folder/'fusion_calibrator.npz') as z:calibrators={k:z[k] for k in z.files if k!='weight'};assert float(z['weight'])==.5
        expected_rows=len(y);recorded_prevalence=None
        for variant in VARIANTS:
            with np.load(folder/variant/'scores.npz',allow_pickle=False) as z:cal,test=z['calibration'],z['test']
            assert len(test)==expected_rows and np.isfinite(cal).all() and np.isfinite(test).all()
            if variant=='baseline':
                np.testing.assert_allclose(cal,baseline['calibration'],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(test,baseline['test'],rtol=1e-12,atol=1e-12)
            else:
                name=variant.split('_')[0];ref=calibrators[name];ranked=-np.log((len(ref)+1-np.searchsorted(ref,ref,side='left'))/(len(ref)+1))
                implied=2*cal-saved_baseline['calibration'] if variant.endswith('_fused') else cal
                np.testing.assert_allclose(np.sort(implied),ranked,atol=1e-12,rtol=1e-12)
                if variant in ['cnn_fused','kan_fused']:
                    with np.load(folder/(name+'_only')/'scores.npz') as z:
                        np.testing.assert_array_equal(cal,.5*saved_baseline['calibration']+.5*z['calibration'])
                        np.testing.assert_array_equal(test,.5*saved_baseline['test']+.5*z['test'])
            th=json.loads((folder/variant/'thresholds.json').read_text());r=next(r for r in records if r['dataset']==dataset and r['seed']==seed and r['variant']==variant);m=r['metrics'];s=test[known]
            pr,re,_=precision_recall_curve(target,s)
            for k,v in dict(evaluated=len(target),positives=int((target==1).sum()),normal=int((target==0).sum()),unknown_excluded=int((~known).sum()),prevalence=float(target.mean()),average_precision=average_precision_score(target,s),roc_auc=roc_auc_score(target,s),pr_auc_trapezoid=auc(re,pr)).items():assertclose(m[k],v)
            for op,t in th.items():
                if t['status']!='ok':assert m['operating_points'][op] is None;continue
                if op=='quantile':value=np.quantile(cal,.99)
                else:
                    u=np.quantile(cal,.85);peaks=cal[cal>u]-u;shape,_,scale=genpareto.fit(peaks,floc=0);value=u+genpareto.isf(.01*len(cal)/len(peaks),shape,scale=scale)
                assertclose(t['threshold'],value)
                alarm=s>=value;tp=int(np.sum(alarm&(target==1)));fp=int(np.sum(alarm&(target==0)));fn=int(np.sum(~alarm&(target==1)));tn=int(np.sum(~alarm&(target==0)))
                vals=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),fpr=fp/(fp+tn),alarm_fraction=float(alarm.mean()))
                for k,v in vals.items():assertclose(m['operating_points'][op][k],v)
            if variant=='baseline':
                old=next(r for r in historic if r['dataset']==dataset and r['seed']==seed and r['variant']=='cm' and r['scoring']=='tail_flat')
                assert m==old['metrics'],'Historical baseline metrics changed'
        checks.append(dict(dataset=dataset,seed=seed,independent_baseline_bitwise_equal=baseline_exact,independent_baseline_max_abs=baseline_delta,independent_baseline_tolerance=dict(rtol=1e-12,atol=1e-12),historical_baseline_metrics_exact=True,score_variants=9,target_sha256=sha(a.data/dataset/'test.y.i8')))
    write(a.output,dict(status='passed',groups=9,configurations=81,pinned_code_commit=commit,paired_control_rng=not a.pilot,pilot_only=a.pilot,runtime_source_hashes_match_pinned_commit=True,historical_baseline_metrics_exact=True,independent_score_fusion_tail_and_metric_replay=True,all_recorded_pre_label_hashes_verified=True,checks=checks))
    print('AUDIT PASSED:9 paired backbones,81 configs,162 thresholds,all old baseline metrics exact')


if __name__=='__main__':main()
