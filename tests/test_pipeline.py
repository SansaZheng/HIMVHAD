import json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from himvhad.data import demo_data,prepare_csv,load_data
from himvhad.runner import run_group,rescore
from himvhad.model import HIMVHAnomalyDetector
from himvhad.scoring import SkillScorer
from himvhad.thresholds import calibrate
from himvhad.evaluation import metrics

@pytest.mark.parametrize('variant',['legacy','f00','c','m','cm'])
def test_pipeline_without_fit_labels_and_all_gates_before_test(tmp_path,monkeypatch,variant):
    data=tmp_path/'data';demo_data(data);out=tmp_path/'run';original=np.fromfile
    def guarded(file,*args,**kwargs):
        name=Path(file).name
        assert name not in ['train.y.i8','calibration.y.i8']
        if name=='test.y.i8':
            for score in ['field','flat','tail']:assert (out/score/'label_access_gate.json').exists()
        return original(file,*args,**kwargs)
    monkeypatch.setattr(np,'fromfile',guarded)
    r=run_group(data,out,variant,steps=2,hidden=16,scores=['field','flat','tail'])
    assert len(r)==3 and all(x['metrics']['evaluated']==128 for x in r)
    with pytest.raises(FileExistsError):run_group(data,out,steps=1)

def test_no_evaluation_labels_still_scores(tmp_path):
    data=tmp_path/'data';demo_data(data);(data/'test.y.i8').unlink();r=run_group(data,tmp_path/'run',steps=1,hidden=16,scores=['tail'])
    assert r[0]['status']=='scored_without_test_labels' and r[0]['metrics'] is None

def test_skill_reload_and_checkpoint_generic_schema(tmp_path):
    data=tmp_path/'data';demo_data(data);out=tmp_path/'run';r=run_group(data,out,steps=2,hidden=16,scores=['skill_tail','skill_raw']);obj=SkillScorer.load(out/'skill_calibrator.npz')
    with np.load(out/'components.npz') as z:values=obj.all_scores(z['test'])
    with np.load(out/'skill_tail/scores.npz') as z:np.testing.assert_array_equal(values['skill_tail'],z['test'])
    model=HIMVHAnomalyDetector.load(out/'model.pt');assert model.width==6 and model.hidden==16 and len(model.groups)==3

def test_rescore_does_not_retrain(tmp_path,monkeypatch):
    data=tmp_path/'data';demo_data(data);out=tmp_path/'run';run_group(data,out,steps=1,hidden=16,scores=['field','tail'])
    def forbidden(*a,**k):raise AssertionError('rescore must not call fit')
    monkeypatch.setattr(HIMVHAnomalyDetector,'fit',forbidden);rescore(out,tmp_path/'scores',['field','tail'],data)
    for name in ['field','tail']:
        with np.load(out/name/'scores.npz') as a,np.load(tmp_path/'scores'/name/'scores.npz') as b:
            for p in ['calibration','test']:np.testing.assert_array_equal(a[p],b[p])

def test_csv_no_label_features_and_whole_time_splits(tmp_path):
    import pandas as pd
    t=np.repeat(np.arange(30),2);df=pd.DataFrame(dict(time=t,sender=t%4,target=t%3,amount=t+1,kind=t%2,label=t%2));df=df.iloc[::-1];csv=tmp_path/'input.csv';df.to_csv(csv,index=False)
    config=dict(dataset='custom',time='time',views=['sender','target','kind'],numeric=['amount'],categorical={'kind':4},label='label',signed_log=['amount']);f=tmp_path/'config.json';f.write_text(json.dumps(config));out=tmp_path/'data';schema=prepare_csv(csv,f,out);assert 'label' not in schema['features']
    assert not (out/'train.y.i8').exists() and not (out/'calibration.y.i8').exists();r=run_group(out,tmp_path/'run',steps=1,hidden=16,scores=['flat']);assert r[0]['metrics']['evaluated']>0
    config['views'].append('label');f.write_text(json.dumps(config))
    with pytest.raises(ValueError):prepare_csv(csv,f,tmp_path/'bad')

def test_invalid_field_partition_is_rejected():
    with pytest.raises(ValueError):HIMVHAnomalyDetector(3,'custom',field_groups=[[0,1],[1,2]])

def test_unknown_targets_and_degenerate_pot():
    c=np.zeros(40);th=calibrate(c);assert th['pot']['status']=='failed';m=metrics(np.array([0,1,-1]),np.array([0.,1.,100.]),th);assert m['unknown_excluded']==1 and m['evaluated']==2
    assert m['operating_points']['pot'] is None

def test_cli_demo_is_separate_from_financial_evidence(tmp_path):
    out=tmp_path/'demo';r=subprocess.run([sys.executable,'-m','himvhad','demo','--output',str(out),'--steps','1'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr;assert json.loads((out/'DEMO_ONLY.json').read_text())['financial_benchmark'] is False

def test_empty_native_views_and_checkpoint_roundtrip(tmp_path):
    from himvhad.network import hyperedges
    rng=np.random.default_rng(3);x=rng.normal(size=(8,3)).astype('f4');views=[hyperedges(np.arange(8),4) for _ in range(2)]
    m=HIMVHAnomalyDetector(3,'custom',steps=1,field_groups=[[0],[1,2]],nviews=2,hidden=8).fit(x,views);m.save(tmp_path/'m.pt');n=HIMVHAnomalyDetector.load(tmp_path/'m.pt')
    np.testing.assert_array_equal(m.components(x,views),n.components(x,views));assert np.isfinite(n.score(x,views)).all()

def test_experimental_rescore_with_original_data(tmp_path):
    data=tmp_path/'data';demo_data(data);out=tmp_path/'model';run_group(data,out,steps=1,hidden=16,scores=['skill_tail'])
    rescore(out,tmp_path/'new',['skill_tail'],data)
    with np.load(out/'skill_tail/scores.npz') as a,np.load(tmp_path/'new/skill_tail/scores.npz') as b:
        np.testing.assert_array_equal(a['test'],b['test'])
