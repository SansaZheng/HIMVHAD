"""Fit without labels, freeze all scores/thresholds, then optional evaluation."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .data import PHASES,load_data,load_views
from .model import HIMVHAnomalyDetector
from .network import HIMVHZero,hyperedges,discrepancies
from .preprocessing import Preprocessor
from .scoring import FieldScorer,SkillScorer
from .thresholds import calibrate
from .evaluation import metrics
from .io import sha,write,provenance

SCORES=('field','flat','tail','tail_flat','mean_scale','median_scale','tail_max','tail_corr','skill_tail','skill_raw','skill_equal_tail','skill_hard_tail')


def _reload_probe(model,loaded,x):
    for k,v in model.net.state_dict().items():
        if not torch.equal(v.detach().cpu(),loaded.net.state_dict()[k].detach().cpu()):raise AssertionError('Checkpoint tensor mismatch: '+k)
    rows=x[:min(64,len(x))];v=[hyperedges(np.arange(len(rows))//8) for _ in range(model.nviews if hasattr(model,'nviews') else 4)]
    a=copy.deepcopy(model.net).cpu().eval();b=copy.deepcopy(loaded.net).cpu().eval();enh=torch.as_tensor(discrepancies(rows,v));e=[torch.as_tensor(t) for t in v]
    with torch.no_grad():first=a(enh,e).numpy();again=b(enh,e).numpy()
    np.testing.assert_array_equal(first,again);return float(np.max(np.abs(first-again)))


def run_group(data,out,variant='cm',seed=42,steps=None,scores=('field','flat','tail','tail_flat'),device='cpu',hidden=256,max_train=8192,max_calibration=100000):
    data,out=Path(data),Path(out)
    if out.exists():raise FileExistsError('Use a fresh run output directory')
    if variant not in ['legacy','f00','c','m','cm']:raise ValueError('Unknown context/masking variant')
    if not scores or any(s not in SCORES for s in scores) or len(set(scores))!=len(scores):raise ValueError('Invalid or duplicate score names')
    if min(max_train,max_calibration)<2:raise ValueError('Sampling limits must be >=2')
    schema,raw=load_data(data);nviews=len(schema['view_columns']);width=len(schema['features']);out.mkdir(parents=True);write(out/'input_schema.json',dict(features=schema['features'],field_groups=schema['field_groups']));start=time.monotonic()
    files=[data/'schema.json']+[data/f'{s}.{ext}' for s in PHASES for ext in ['x.f32','keys.npz']]
    write(out/'run_manifest.json',dict(**provenance(),dataset=schema['dataset'],variant=variant,seed=seed,steps=steps if steps is not None else (200 if variant=='legacy' else 800),score_names=list(scores),device=device,max_train=max_train,max_calibration=max_calibration,hidden=hidden,input_sha256={p.name:sha(p) for p in files},zero_real_labels=True))
    ti=np.sort(np.random.default_rng(seed).choice(len(raw['train']),min(max_train,len(raw['train'])),replace=False));ci=np.sort(np.random.default_rng(seed+100).choice(len(raw['calibration']),min(max_calibration,len(raw['calibration'])),replace=False))
    pre=Preprocessor().fit(raw['train'][ti]);pre.save(out/'preprocessor.npz');x={s:pre.transform(raw[s]) for s in PHASES}
    if x['train'].shape[1]!=width:raise ValueError('Prepared features must be finite with a fixed field schema')
    np.savez(out/'indices.npz',train=ti,calibration=ci,test=np.arange(len(raw['test'])));views={s:load_views(data,s) for s in PHASES};steps=steps if steps is not None else (200 if variant=='legacy' else 800)
    torch.set_num_threads(4)
    if device.startswith('cuda'):
        if not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    with threadpool_limits(limits=4):
        if variant=='legacy':
            if any(s.startswith('skill_') for s in scores):raise ValueError('Skill scoring needs semantic masked-field components; use m/cm')
            model=HIMVHZero(width,nviews,seed,device,steps,hidden).fit(x['train'][ti],load_views(data,'train',ti));model.save(out/'model.pt');loaded=HIMVHZero.load(out/'model.pt',device)
            probe=_reload_probe(model,loaded,x['train'][ti]);components={s:(model.score(x[s],views[s])[ci] if s=='calibration' else model.score(x[s],views[s]))[:,None] for s in ['calibration','test']};widths=[width]
        else:
            native=variant in ['c','cm'];consistent=variant in ['m','cm']
            model=HIMVHAnomalyDetector(width,schema['dataset'],seed,device,native,consistent,steps,field_groups=schema['field_groups'],nviews=nviews,hidden=hidden)
            model.fit(x['train'] if native else x['train'][ti],views['train'] if native else load_views(data,'train',ti),ti if native else np.arange(len(ti)),None if native else ti)
            model.save(out/'model.pt');loaded=HIMVHAnomalyDetector.load(out/'model.pt',device);probe=_reload_probe(model,loaded,x['train'][ti]);components={s:(model.components(x[s],views[s])[ci] if s=='calibration' else model.components(x[s],views[s])) for s in ['calibration','test']};widths=[len(g) for g in schema['field_groups']]
    np.savez_compressed(out/'components.npz',**components);calibrator=FieldScorer().fit(components['calibration'],widths);calibrator.save(out/'field_calibrator.npz');computed={s:calibrator.all_scores(c) for s,c in components.items()}
    if any(s.startswith('skill_') for s in scores):
        skill=SkillScorer().fit(x['train'][ti],x['calibration'][ci],components['calibration'],[np.array(g) for g in schema['field_groups']]);skill.save(out/'skill_calibrator.npz')
        for s,c in components.items():computed[s].update(skill.all_scores(c))
    # Complete every requested score and threshold before parsing any target.
    for name in scores:
        folder=out/name;folder.mkdir();cs,ts=computed['calibration'][name],computed['test'][name]
        if not np.isfinite(cs).all() or not np.isfinite(ts).all():raise FloatingPointError('Nonfinite scores')
        np.savez_compressed(folder/'scores.npz',calibration=cs,test=ts);thresholds=calibrate(cs);write(folder/'thresholds.json',thresholds)
        write(folder/'label_access_gate.json',dict(status='scores_and_thresholds_saved_before_labels',checkpoint_sha256=sha(out/'model.pt'),components_sha256=sha(out/'components.npz'),field_calibrator_sha256=sha(out/'field_calibrator.npz'),skill_calibrator_sha256=sha(out/'skill_calibrator.npz') if (out/'skill_calibrator.npz').exists() else None,scores_sha256=sha(folder/'scores.npz'),thresholds_sha256=sha(folder/'thresholds.json')))
    label=data/'test.y.i8';target=np.fromfile(label,dtype='i1') if label.exists() else None;records=[]
    for name in scores:
        folder=out/name;thresholds=json.loads((folder/'thresholds.json').read_text());record=dict(dataset=schema['dataset'],variant=variant,seed=seed,scoring=name,status='evaluated' if target is not None else 'scored_without_test_labels',metrics=metrics(target,computed['test'][name],thresholds) if target is not None else None,scoring_financial_evidence='experimental_skill_weights_not_original_paper' if name.startswith('skill_') else 'run_specific_no_paper_score_claim',model_info=model.info,reload_cpu_probe_max_abs=probe,checkpoint_sha256=sha(out/'model.pt'),scores_sha256=sha(folder/'scores.npz'),target_sha256=sha(label) if target is not None else None)
        write(folder/'result.json',record);records.append(record)
    write(out/'records.json',records);write(out/'complete.json',dict(status='complete',scoring_settings=len(records),new_network_fits=1,seconds=time.monotonic()-start));return records


def rescore(parent,out,scores=('field','flat','tail','tail_flat'),data=None):
    parent,out=Path(parent),Path(out)
    if out.exists():raise FileExistsError('Use a fresh score output directory')
    if any(s not in SCORES for s in scores):raise ValueError('Unknown scoring method')
    if any(s.startswith('skill_') for s in scores) and data is None:raise ValueError('Skill scoring requires --data with the original training/calibration features')
    with np.load(parent/'components.npz',allow_pickle=False) as z:comp={s:z[s] for s in ['calibration','test']}
    model=HIMVHAnomalyDetector.load(parent/'model.pt','cpu');scorer=FieldScorer().fit(comp['calibration'],[len(g) for g in model.groups]);out.mkdir(parents=True);scorer.save(out/'field_calibrator.npz');values={s:scorer.all_scores(c) for s,c in comp.items()};records=[]
    if any(s.startswith('skill_') for s in scores):
        schema,raw=load_data(data);pre=Preprocessor.load(parent/'preprocessor.npz')
        with np.load(parent/'indices.npz',allow_pickle=False) as z:ti,ci=z['train'],z['calibration']
        skill=SkillScorer().fit(pre.transform(raw['train'][ti]),pre.transform(raw['calibration'][ci]),comp['calibration'],model.groups);skill.save(out/'skill_calibrator.npz')
        for s,c in comp.items():values[s].update(skill.all_scores(c))
    for name in scores:
        folder=out/name;folder.mkdir();np.savez_compressed(folder/'scores.npz',calibration=values['calibration'][name],test=values['test'][name]);thresholds=calibrate(values['calibration'][name]);write(folder/'thresholds.json',thresholds);write(folder/'label_access_gate.json',dict(status='scores_and_thresholds_saved_before_labels',parent_checkpoint_sha256=sha(parent/'model.pt'),parent_components_sha256=sha(parent/'components.npz'),field_calibrator_sha256=sha(out/'field_calibrator.npz'),skill_calibrator_sha256=sha(out/'skill_calibrator.npz') if (out/'skill_calibrator.npz').exists() else None,scores_sha256=sha(folder/'scores.npz'),thresholds_sha256=sha(folder/'thresholds.json')))
    y=np.fromfile(Path(data)/'test.y.i8',dtype='i1') if data is not None else None
    for name in scores:
        folder=out/name;r=dict(scoring=name,new_network_fits=0,metrics=metrics(y,values['test'][name],json.loads((folder/'thresholds.json').read_text())) if y is not None else None);write(folder/'result.json',r);records.append(r)
    write(out/'records.json',records);return records


def predict(run,data,out,score='tail',phase='test'):
    """Apply frozen checkpoint/scaler/calibrator/thresholds; never read targets."""
    run,data,out=Path(run),Path(data),Path(out)
    if out.exists():raise FileExistsError('Use a fresh prediction directory')
    schema,raw=load_data(data);pre=Preprocessor.load(run/'preprocessor.npz');model=HIMVHAnomalyDetector.load(run/'model.pt','cpu');x=pre.transform(raw[phase])
    if x.shape[1]!=model.width or schema['field_groups']!=[g.tolist() for g in model.groups]:raise ValueError('Prepared feature schema differs from checkpoint')
    expected=run/'input_schema.json'
    if expected.exists() and json.loads(expected.read_text())['features']!=schema['features']:raise ValueError('Prepared feature order differs from checkpoint')
    components=model.components(x,load_views(data,phase));calibrator=SkillScorer.load(run/'skill_calibrator.npz') if score.startswith('skill_') else FieldScorer.load(run/'field_calibrator.npz');values=calibrator.all_scores(components)
    if score not in values:raise ValueError('Unsupported scoring method for this calibrator')
    thresholds=json.loads((run/score/'thresholds.json').read_text());out.mkdir(parents=True);np.savez_compressed(out/'predictions.npz',score=values[score],**{k:values[score]>=v['threshold'] for k,v in thresholds.items() if v['status']=='ok'})
    write(out/'prediction_manifest.json',dict(source=provenance(),rows=len(x),phase=phase,scoring=score,checkpoint_sha256=sha(run/'model.pt'),scaler_sha256=sha(run/'preprocessor.npz'),calibrator_sha256=sha(run/('skill_calibrator.npz' if score.startswith('skill_') else 'field_calibrator.npz')),thresholds_sha256=sha(run/score/'thresholds.json'),real_labels_read=False));return values[score]
