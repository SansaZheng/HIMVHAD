import copy
import numpy as np
import pytest
import torch
from himvhad.network import HimvhNet,NoveltyLayer,hyperedges,discrepancies
from himvhad.hypergraph import NativeContext,block_forward,assignments,masked


def test_hyperedges_no_cross_key_and_literal_reverse_kl():
    keys = ['a', 'b', 'a', 'a', 'a', 'b', 'b', 'b']
    edges = hyperedges(keys, 4)
    assert edges.tolist() == [[0, 2, 3, 4], [1, 5, 6, 7]]
    x = np.arange(16, dtype=np.float32).reshape(8, 2)
    features = discrepancies(x, [edges, edges])
    np.testing.assert_array_equal(features[:, -3:], np.zeros((8, 3)))
    assert len(hyperedges(['a', 'a'], 4)) == 0

def test_novelty_recipient_excludes_own_message():
    layer = NoveltyLayer(1).double()
    with torch.no_grad():
        layer.weight.weight.fill_(1)
    h = torch.tensor([[1.], [3.]], dtype=torch.float64)
    actual = layer(h, torch.tensor([[0, 1]]))
    # Both deviations have equal novelty; alpha=.5. Recipient 0 gets .5*3.
    torch.testing.assert_close(actual, torch.sigmoid(torch.tensor([[2.5], [3.5]], dtype=torch.float64)))

def test_native_ego_matches_full_forward_and_parameter_gradients():
    torch.set_num_threads(2);torch.manual_seed(3)
    x=np.random.default_rng(3).normal(size=(20,3)).astype(np.float32)
    views=[hyperedges(np.arange(20)//5),hyperedges(np.arange(20)%4)]*2
    ctx=NativeContext(x,views);enh=ctx.enhanced(x,np.arange(20))
    np.testing.assert_allclose(enh,discrepancies(x,views),atol=1e-7)
    targets=np.array([7,1,14]);net=HimvhNet(3,4,hidden=8,layers=3).eval();other=copy.deepcopy(net)
    dense=net(torch.tensor(enh),[torch.tensor(e) for e in views])[targets]
    plan=ctx.plan(targets);needed=np.unique(np.concatenate([b[0][0] for b in plan]))
    block=block_forward(other,ctx.enhanced(x,needed),plan,targets,'cpu',needed)
    torch.testing.assert_close(dense,block,rtol=1e-5,atol=1e-6)
    dense.square().sum().backward();block.square().sum().backward()
    for (name,a),(n,b) in zip(net.named_parameters(),other.named_parameters()):
        assert name==n;torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-6)

def test_legacy_clean_auxiliary_leaks_masked_value_corrected_aux_does_not():
    x=np.random.default_rng(8).normal(size=(12,3)).astype(np.float32);changed=x.copy();changed[0,0]+=20
    views=[hyperedges(np.arange(12)//4),hyperedges(np.arange(12)%3)]*2
    old_a=discrepancies(x,views);old_b=discrepancies(changed,views)
    old_a[0,0]=old_b[0,0]=0
    assert np.max(np.abs(old_a-old_b))>1e-4  # hidden value remains in clean auxiliary channels
    x[0,0]=changed[0,0]=0
    c=NativeContext(x,views)
    np.testing.assert_array_equal(c.enhanced(x,np.arange(12)),c.enhanced(changed,np.arange(12)))

def test_stateless_masks_preserve_node_identity_and_cover_every_field():
    ids=np.array([4,77,9,12]);a=assignments(ids,42,5)
    np.testing.assert_array_equal(a[[2,0]],assignments(ids[[2,0]],42,5))
    for i in range(4):assert set((a[i]+np.arange(5))%5)==set(range(5))
