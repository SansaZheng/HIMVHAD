import numpy as np
import pytest
from himvhad.scoring import FieldScorer,SkillScorer

def test_empirical_tail_ties_and_bounds(tmp_path):
    c=np.array([[0.,1],[0,2],[2,2],[3,4]])
    o=FieldScorer().fit(c,[1,8]);q=np.array([[0,0],[1,2],[100,100]])
    expected=np.array([[-np.log((1+(c[:,j]>=v).sum())/5) for j,v in enumerate(row)] for row in q])
    np.testing.assert_array_equal(o.evidence(q),expected)
    assert np.max(expected)==np.log(5)
    o.save(tmp_path/'f.npz');loaded=FieldScorer.load(tmp_path/'f.npz')
    for k,v in o.all_scores(q).items():np.testing.assert_array_equal(v,loaded.all_scores(q)[k])

def test_scale_invariance_and_frozen_fit():
    c=np.random.default_rng(0).uniform(.1,8,(100,4));t=c[:10]*2
    a=FieldScorer().fit(c,[1,4,8,16]);b=FieldScorer().fit(c*[2,10,.1,30],[1,4,8,16])
    for k in ['mean_scale','median_scale','tail','tail_corr']:
        np.testing.assert_allclose(a.all_scores(t)[k],b.all_scores(t*[2,10,.1,30])[k],atol=1e-14)
    before=a.sorted.copy();a.all_scores(t*100);np.testing.assert_array_equal(before,a.sorted)

def test_constant_fields_are_safe():
    c=np.zeros((50,3));o=FieldScorer().fit(c,[1,2,3])
    for v in o.all_scores(c).values():assert np.isfinite(v).all();assert (v==0).all()
    np.testing.assert_allclose(o.corr_weights,np.ones(3)/3)
    assert o.evidence([[1,0,0]])[0,0]==np.log(51)
    assert FieldScorer().fit(c[:,:1],[1]).corr_weights[0]==1

@pytest.mark.parametrize('x',[np.array([1]),np.array([[np.nan]]),np.array([[-1.]])])
def test_invalid_field_errors(x):
    with pytest.raises(ValueError):FieldScorer().fit(x,[1])
