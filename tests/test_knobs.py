import json
from pathlib import Path
import numpy as np
import pytest
import torch
from himvhad.knobs import (KnobConfig,KnobAdapter,KANBoundary,SplineEdges,ScoreFusion,
                          hidden_states,training_components,pseudo_split)
from himvhad.model import HIMVHAnomalyDetector
from himvhad.network import hyperedges


def toy():
    torch.set_num_threads(2)
    x=np.random.default_rng(7).normal(size=(32,4)).astype('f4')
    views=[hyperedges(np.arange(32)//8),hyperedges(np.arange(32)%4)]
    model=HIMVHAnomalyDetector(4,'custom',steps=1,field_groups=[[0],[1,2],[3]],nviews=2,hidden=8,layers=2)
    return model,x,views


def test_hidden_fused_state_reconstructs_exact_parent_decoder_and_order():
    model,x,views=toy();before={k:v.clone() for k,v in model.net.state_dict().items()}
    state=hidden_states(model,x,views)
    from himvhad.hypergraph import NativeContext
    ctx=NativeContext(x,views)
    with torch.no_grad():
        dense=model.net(torch.tensor(ctx.enhanced(x,np.arange(len(x)))),[torch.tensor(e) for e in views])
        reconstructed=model.net.decoder(torch.tensor(state[:,-1]))
    torch.testing.assert_close(dense,reconstructed)
    rows=np.array([23,1,9]);np.testing.assert_allclose(hidden_states(model,x,views,rows),state[rows],atol=1e-7)
    for name,value in model.net.state_dict().items():assert torch.equal(value,before[name])


def test_training_proxy_uses_exact_masked_reconstruction_and_identity():
    model,x,views=toy();rows=np.array([23,1,9,4])
    actual=training_components(model,x,views,rows,batch=2)
    expected=model.components(x,views)[rows]
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)


def test_pseudo_thresholds_do_not_use_internal_validation_values():
    cfg=KnobConfig();error=np.linspace(0,1,100)
    y,fit,valid,th=pseudo_split(error,42,cfg)
    rng=np.random.default_rng(42+710);order=rng.permutation(100);error[order[80:]]=error[order[80:][::-1]]
    _,_,_,changed=pseudo_split(error,42,cfg)
    assert th==changed and not set(fit)&set(valid)
    with pytest.raises(ValueError):pseudo_split(np.ones(100),42,cfg)


def test_spline_partition_and_closed_form_edges_have_correct_gradient():
    layer=SplineEdges(2,1,9).double()
    with torch.no_grad():
        layer.linear.weight.zero_();layer.linear.bias.zero_()
        # Linear knot coefficients must reproduce x0 +2*x1 inside grid.
        layer.coeff[0,0]=layer.knots;layer.coeff[0,1]=2*layer.knots
    z=torch.tensor([[.3,-.7],[-1.2,2.4]],dtype=torch.double,requires_grad=True)
    value=layer(z).squeeze(1)
    torch.testing.assert_close(value,z[:,0]+2*z[:,1])
    torch.testing.assert_close(torch.autograd.grad(value.sum(),z)[0],torch.tensor([[1.,2.],[1.,2.]],dtype=torch.double))


def test_adapter_roundtrip_internal_selection_and_interventions(tmp_path):
    torch.set_num_threads(2);rng=np.random.default_rng(10)
    h=rng.normal(size=(128,4,8)).astype('f4');errors=np.exp(h[:,0,0])
    cfg=KnobConfig(probe_steps=3,student_steps=3,max_knobs=3,channels=2)
    model=KnobAdapter(cfg).fit(h,errors);model.save(tmp_path/'adapter.pt')
    restored=KnobAdapter.load(tmp_path/'adapter.pt')
    for name,value in model.logits(h[:12]).items():
        np.testing.assert_array_equal(value,restored.logits(h[:12])[name])
    assert set(model.info['fit_rows']).isdisjoint(model.info['validation_rows'])
    assert model.info['selected_k']==model.info['sweep'][-1]['k']
    for record in model.intervention_diagnostics(h[:8])['records']:
        if record['status']=='ok':assert record['pooled_patch_identity_max_abs']<1e-5
    if model.info['selected_k']==1:
        assert any(r['status']=='undefined_in_one_dimension' for r in model.intervention_diagnostics(h[:8])['records'])
    with pytest.raises(ValueError):restored.logits(h[:12,:,:7])


def test_fusion_is_calibration_only_baseline_unchanged_and_serializable(tmp_path):
    cal={'cnn':np.linspace(-2,2,100),'kan':np.zeros(100)}
    fusion=ScoreFusion(.5).fit(cal);fusion.save(tmp_path/'fusion.npz');loaded=ScoreFusion.load(tmp_path/'fusion.npz')
    baseline=np.array([.1,1.,2.]);logits={'cnn':np.array([-1.,0.,1.]),'kan':np.ones(3)}
    result=fusion.scores(baseline,logits)
    np.testing.assert_array_equal(result['baseline'],baseline)
    for name,value in result.items():np.testing.assert_array_equal(value,loaded.scores(baseline,logits)[name])
    # Changing other test rows never changes the first row's prediction.
    logits['cnn'][1:]=999
    assert fusion.scores(baseline,logits)['cnn_fused'][0]==result['cnn_fused'][0]


def test_knob_pipeline_never_fits_backbone_and_gates_all_labels(tmp_path,monkeypatch):
    from himvhad.data import demo_data
    from himvhad.runner import run_group
    from himvhad.knob_runner import run_knobs,evaluate_knobs,predict_knobs,VARIANTS
    data=tmp_path/'data';parent=tmp_path/'parent';out=tmp_path/'knobs';demo_data(data)
    run_group(data,parent,steps=1,hidden=8,scores=['tail_flat'])
    original=np.fromfile
    def forbidden(*a,**k):raise AssertionError('Backbone must stay frozen')
    monkeypatch.setattr(HIMVHAnomalyDetector,'fit',forbidden)
    def guard(file,*a,**kw):
        name=Path(file).name
        assert name not in ['train.y.i8','calibration.y.i8']
        if name=='test.y.i8':
            assert (out/'label_access_gate.json').exists()
            for method in VARIANTS:assert (out/method/'thresholds.json').exists()
        return original(file,*a,**kw)
    monkeypatch.setattr(np,'fromfile',guard)
    cfg=KnobConfig(probe_steps=2,student_steps=2,max_knobs=2,channels=2)
    run_knobs(parent,data,out,cfg,evaluate=False);records=evaluate_knobs(out,data)
    assert len(records)==9
    with np.load(parent/'tail_flat/scores.npz') as a,np.load(out/'baseline/scores.npz') as b:
        np.testing.assert_array_equal(a['test'],b['test']);np.testing.assert_array_equal(a['calibration'],b['calibration'])
    def no_labels(file,*a,**kw):
        assert not str(file).endswith('.y.i8');return original(file,*a,**kw)
    monkeypatch.setattr(np,'fromfile',no_labels)
    values=predict_knobs(parent,out,data,tmp_path/'prediction')
    for method in VARIANTS:
        with np.load(out/method/'scores.npz') as z:np.testing.assert_array_equal(values[method],z['test'])
    schema=json.loads((data/'schema.json').read_text());schema['features'].reverse();(data/'schema.json').write_text(json.dumps(schema))
    with pytest.raises(ValueError):predict_knobs(parent,out,data,tmp_path/'wrong')
    with (out/'kan_fused/scores.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(AssertionError):evaluate_knobs(out,data)
