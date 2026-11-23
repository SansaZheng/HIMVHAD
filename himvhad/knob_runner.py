"""Paired frozen-backbone experiments, with score artifacts before labels."""
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .knobs import KnobAdapter, KnobConfig, ScoreFusion, hidden_states, training_components
from .data import load_data, load_views, PHASES
from .model import HIMVHAnomalyDetector
from .preprocessing import Preprocessor
from .scoring import FieldScorer
from .thresholds import calibrate
from .evaluation import metrics
from .io import write, sha, provenance

VARIANTS = ('baseline', 'cnn_only', 'cnn_fused', 'kan_only', 'kan_fused',
            'random_fused', 'uniform_fused', 'linear_fused', 'symbolic_fused')


def run_knobs(parent, data, out, config=None, device='cpu', evaluate=True):
    parent, data, out = Path(parent), Path(data), Path(out)
    if out.exists():raise FileExistsError('Use a fresh knob experiment output')
    config = config or KnobConfig(); config.validate()
    schema, raw = load_data(data)
    model = HIMVHAnomalyDetector.load(parent/'model.pt', device)
    if not model.native or not model.consistent:
        raise ValueError('This paired protocol requires the frozen cm backbone')
    if model.width != len(schema['features']) or [g.tolist() for g in model.groups] != schema['field_groups']:
        raise ValueError('Parent feature groups do not match the prepared data')
    pre = Preprocessor.load(parent/'preprocessor.npz')
    with np.load(parent/'indices.npz', allow_pickle=False) as z:
        ti, ci, test = z['train'], z['calibration'], z['test']
    np.testing.assert_array_equal(test, np.arange(len(raw['test'])))
    expected_ti = np.sort(np.random.default_rng(model.seed).choice(len(raw['train']), min(8192,len(raw['train'])),replace=False))
    expected_ci = np.sort(np.random.default_rng(model.seed+100).choice(len(raw['calibration']), min(100000,len(raw['calibration'])),replace=False))
    np.testing.assert_array_equal(ti, expected_ti);np.testing.assert_array_equal(ci, expected_ci)
    with np.load(parent/'components.npz', allow_pickle=False) as z:
        comp = {s:z[s] for s in ['calibration','test']}
    if len(comp['calibration'])!=len(ci) or len(comp['test'])!=len(test):
        raise ValueError('Parent component rows differ from canonical partitions')
    out.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(4)
    parent_hashes = {name:sha(parent/name) for name in ['model.pt','preprocessor.npz','indices.npz','components.npz']}
    input_hashes = {p.name:sha(p) for p in [data/'schema.json']+[data/f'{s}.{ext}' for s in PHASES for ext in ['x.f32','keys.npz']]}
    state_before={k:v.detach().cpu().clone() for k,v in model.net.state_dict().items()}
    if device.startswith('cuda'):
        if not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.cuda.reset_peak_memory_stats()
    write(out/'run_manifest.json',dict(source=provenance(),config=vars(config),dataset=schema['dataset'],seed=model.seed,
          parent_sha256=parent_hashes,input_sha256=input_hashes,device=device,zero_real_labels=True,
          baseline='unaltered parent cm_tail_flat',features=schema['features'],new_backbone_fits=0))
    field=FieldScorer().fit(comp['calibration'],[len(g) for g in model.groups])
    baseline={s:field.all_scores(comp[s])['tail_flat'] for s in comp}
    field.save(out/'baseline_calibrator.npz')
    with threadpool_limits(limits=4):
        x=pre.transform(raw['train']);views=load_views(data,'train')
        states=hidden_states(model,x,views,ti)
        errors=training_components(model,x,views,ti)
        training_target=np.average(errors,axis=1,weights=[len(g) for g in model.groups])
        np.savez_compressed(out/'training_proxy.npz',indices=ti,components=errors,target=training_target)
        adapter=KnobAdapter(config,model.seed,device).fit(states,training_target)
        adapter.save(out/'adapter.pt');write(out/'adapter_info.json',adapter.info)
        validation=np.asarray(adapter.info['validation_rows'])[:128]
        write(out/'interventions.json',adapter.intervention_diagnostics(states[validation]))
        # Restore check: prediction, selection and symbolic coefficients survive serialization.
        loaded=KnobAdapter.load(out/'adapter.pt',device)
        original,restored=adapter.logits(states[:32]),loaded.logits(states[:32])
        for name in original:np.testing.assert_array_equal(original[name],restored[name])
        del x,views,states,loaded;gc.collect()
        proxy={}
        for phase,rows in [('calibration',ci),('test',test)]:
            x=pre.transform(raw[phase]);views=load_views(data,phase)
            states=hidden_states(model,x,views,rows)
            proxy[phase]=adapter.logits(states)
            del x,views,states;gc.collect()
        fusion=ScoreFusion(config.fusion).fit(proxy['calibration']);fusion.save(out/'fusion_calibrator.npz')
        values={s:fusion.scores(baseline[s],proxy[s]) for s in proxy}
    for name in VARIANTS:
        folder=out/name;folder.mkdir()
        np.savez_compressed(folder/'scores.npz',calibration=values['calibration'][name],test=values['test'][name])
        write(folder/'thresholds.json',calibrate(values['calibration'][name]))
    for name,value in model.net.state_dict().items():
        if not torch.equal(value.detach().cpu(),state_before[name]):
            raise AssertionError('Frozen backbone changed: '+name)
    files=[p for p in out.rglob('*') if p.is_file()]
    write(out/'label_access_gate.json',dict(status='all_scores_thresholds_and_models_saved_before_test_labels',
          files_sha256={str(p.relative_to(out)):sha(p) for p in sorted(files)},
          baseline_components_preserved=True,backbone_tensors_unchanged=True,
          real_training_calibration_labels_read=False))
    write(out/'scoring_complete.json',dict(status='complete',dataset=schema['dataset'],seed=model.seed,
          score_variants=len(VARIANTS),new_backbone_fits=0,seconds=time.monotonic()-start,
          peak_cuda_allocated_gb=torch.cuda.max_memory_allocated()/2**30 if device.startswith('cuda') else None,
          gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None))
    if evaluate:return evaluate_knobs(out,data)
    return out


def evaluate_knobs(out,data):
    out,data=Path(out),Path(data)
    gate=json.loads((out/'label_access_gate.json').read_text())
    for name,h in gate['files_sha256'].items():
        if sha(out/name)!=h:raise AssertionError('Scoring artifact changed after gate: '+name)
    label=data/'test.y.i8'
    if not label.exists():return []
    y=np.fromfile(label,dtype='i1');manifest=json.loads((out/'run_manifest.json').read_text());records=[]
    for name in VARIANTS:
        with np.load(out/name/'scores.npz',allow_pickle=False) as z:s=z['test']
        thresholds=json.loads((out/name/'thresholds.json').read_text())
        record=dict(dataset=manifest['dataset'],seed=manifest['seed'],variant=name,
                    metrics=metrics(y,s,thresholds),target_sha256=sha(label),
                    scores_sha256=sha(out/name/'scores.npz'),gate_sha256=sha(out/'label_access_gate.json'),
                    parent_sha256=manifest['parent_sha256'],new_backbone_fits=0)
        write(out/name/'result.json',record);records.append(record)
    write(out/'records.json',records);return records


def predict_knobs(parent, adapter_run, data, out, phase='test'):
    """Frozen fitted adapter inference; does not access target arrays."""
    parent,adapter_run,data,out=map(Path,[parent,adapter_run,data,out])
    if out.exists():raise FileExistsError('Use a fresh prediction directory')
    gate=json.loads((adapter_run/'label_access_gate.json').read_text())
    for name in ['adapter.pt','fusion_calibrator.npz','baseline_calibrator.npz']:
        if sha(adapter_run/name)!=gate['files_sha256'][name]:raise ValueError('Adapter file modified')
    manifest=json.loads((adapter_run/'run_manifest.json').read_text())
    for name in ['model.pt','preprocessor.npz']:
        if sha(parent/name)!=manifest['parent_sha256'][name]:raise ValueError('Wrong frozen parent')
    schema,raw=load_data(data);model=HIMVHAnomalyDetector.load(parent/'model.pt');pre=Preprocessor.load(parent/'preprocessor.npz')
    if schema['field_groups']!=[g.tolist() for g in model.groups]:raise ValueError('Field schema mismatch')
    if schema['features']!=manifest['features']:raise ValueError('Feature order differs from fitted adapter')
    x=pre.transform(raw[phase]);views=load_views(data,phase)
    adapter=KnobAdapter.load(adapter_run/'adapter.pt');fusion=ScoreFusion.load(adapter_run/'fusion_calibrator.npz')
    baseline=FieldScorer.load(adapter_run/'baseline_calibrator.npz').all_scores(model.components(x,views))['tail_flat']
    values=fusion.scores(baseline,adapter.logits(hidden_states(model,x,views)))
    out.mkdir(parents=True)
    for name in VARIANTS:
        th=json.loads((adapter_run/name/'thresholds.json').read_text())
        np.savez_compressed(out/f'{name}.npz',score=values[name],**{k:values[name]>=v['threshold'] for k,v in th.items() if v['status']=='ok'})
    write(out/'prediction_manifest.json',dict(source=provenance(),real_labels_read=False,rows=len(x),phase=phase,
          parent_sha256=sha(parent/'model.pt'),adapter_sha256=sha(adapter_run/'adapter.pt')))
    return values
