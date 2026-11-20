"""Native per-view exact ego training and consistent masked-field detection."""
import numpy as np
import torch
from .network import HimvhNet
from .hypergraph import NativeContext, block_forward, assignments, masked, feature_groups


class HIMVHAnomalyDetector:
    def __init__(self, width, task, seed=42, device='cpu', native=True, consistent=True, steps=800, field_groups=None, nviews=4, hidden=256, layers=3):
        self.width, self.task, self.seed, self.device = width, task, seed, device
        self.native, self.consistent, self.steps = native, consistent, steps
        self.groups = feature_groups(task, width) if field_groups is None else [np.asarray(g,dtype=np.int64) for g in field_groups]
        if width<1 or nviews<2 or hidden<1 or layers<1 or steps<1: raise ValueError("Positive model dimensions/steps and >=2 views required")
        if not self.groups or not np.array_equal(np.sort(np.concatenate(self.groups)),np.arange(width)): raise ValueError("Field groups must partition all columns exactly")
        self.nviews,self.hidden,self.layers=nviews,hidden,layers
        torch.manual_seed(seed)
        self.net = HimvhNet(width, nviews, hidden, layers).to(device)
        self.info = dict(native_context=native, consistent_masked_field=consistent, steps=steps, batch=128,
                         target_exposures=steps*128, hidden=hidden, layers=layers, window=4, lr=.001,
                         decoder_normalization='original BatchNorm', supervised_labels=False)

    def fit(self, x, views, target_rows=None, node_ids=None):
        x=self._validate(x,views)
        target_rows=np.arange(len(x)) if target_rows is None else np.asarray(target_rows,dtype=np.int64)
        if target_rows.ndim!=1 or len(target_rows)<2 or len(np.unique(target_rows))!=len(target_rows) or (target_rows<0).any() or (target_rows>=len(x)).any(): raise ValueError("Need >=2 distinct valid training targets")
        context = NativeContext(x, views, node_ids)
        clean = None if self.consistent else context.enhanced(x, np.arange(len(x)))
        opt = torch.optim.Adam(self.net.parameters(), lr=.001)
        rng = np.random.default_rng(self.seed+1234)
        losses = []
        self.net.train()
        for step in range(self.steps):
            targets = rng.choice(target_rows, size=min(128, len(target_rows)), replace=False)
            plan = context.plan(targets,layers=self.layers)
            needed = np.unique(np.concatenate([b[0][0] for b in plan]))
            salt = self.seed*1000003+step*10007
            if self.consistent:
                fields = assignments(context.ids, salt, len(self.groups))
                value = masked(x, fields, self.groups)
                enhanced = context.enhanced(value, needed)
            else:
                enhanced = clean[needed].copy()
                # Same20% marginal raw-coordinate corruption, with deterministic
                # node/feature hashing so context variants share target masks.
                keys = (context.ids[needed, None]*self.width+np.arange(self.width)).reshape(-1)
                mask = assignments(keys, salt, 5).reshape(len(needed), -1) == 0
                enhanced[:, :self.width][mask] = 0
            opt.zero_grad()
            prediction = block_forward(self.net, enhanced, plan, targets, self.device, needed)
            truth = torch.as_tensor(x[targets], device=self.device)
            if self.consistent:
                error = (prediction-truth).square()
                per_row = torch.zeros(len(targets), device=self.device)
                for g, cols in enumerate(self.groups):
                    rows = np.flatnonzero(fields[targets] == g)
                    if len(rows):
                        ti = torch.as_tensor(rows, device=self.device)
                        per_row[ti] = error[ti][:, cols].mean(1)
                loss = per_row.mean()
            else:
                loss = (prediction-truth).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError('HIMVH repair nonfinite loss')
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        self.info.update(loss_first=losses[0], loss_last=losses[-1], target_exposures=self.steps*min(128,len(target_rows)), context_rows=len(x),
                         train_targets=len(target_rows), hyperedges=[len(e) for e in views])
        return self

    def components(self, x, views, node_ids=None):
        x=self._validate(x,views)
        ctx = NativeContext(x, views, node_ids)
        nodes = np.arange(len(x)); result = np.zeros((len(x), len(self.groups)), dtype=np.float64)
        self.net.eval()
        edges = [torch.as_tensor(e, device=self.device) for e in views]
        base = assignments(ctx.ids, self.seed+991, len(self.groups))
        passes = len(self.groups) if self.consistent else 1
        with torch.no_grad():
            for turn in range(passes):
                fields = (base+turn) % len(self.groups)
                value = masked(x, fields, self.groups) if self.consistent else x
                enhanced = torch.as_tensor(ctx.enhanced(value, nodes), device=self.device)
                predicted = self.net(enhanced, edges).cpu().numpy()
                error = (predicted-x)**2
                for g, cols in enumerate(self.groups):
                    rows = fields == g if self.consistent else np.ones(len(x), bool)
                    result[rows, g] = error[rows][:, cols].mean(1)
        return result

    def _validate(self,x,views):
        x=np.asarray(x,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=self.width or not len(x) or not np.isfinite(x).all(): raise ValueError("Need finite Nxwidth features")
        if len(views)!=self.nviews: raise ValueError("Wrong view count")
        for e in views:
            e=np.asarray(e)
            if e.ndim!=2 or e.dtype.kind not in "iu" or (e<0).any() or (e>=len(x)).any(): raise ValueError("Invalid hyperedges")
        return x

    def score(self,x,views,flat=False,node_ids=None):
        return self.score_components(self.components(x,views,node_ids),flat=flat)

    def score_components(self, comp, flat=False):
        if flat or not self.consistent:
            return np.average(comp, axis=1, weights=[len(c) for c in self.groups])
        return comp.mean(1)

    def save(self, path):
        torch.save(dict(width=self.width, task=self.task, seed=self.seed, native=self.native, consistent=self.consistent,
                        steps=self.steps, field_groups=[g.tolist() for g in self.groups], nviews=self.nviews, hidden=self.hidden, layers=self.layers, format_version=1, state=self.net.state_dict(), info=self.info), path)

    @classmethod
    def load(cls, path, device='cpu'):
        d = torch.load(path, map_location=device, weights_only=True)
        model = cls(d['width'], d['task'], d['seed'], device, d['native'], d['consistent'], d['steps'], field_groups=d.get('field_groups'), nviews=d.get('nviews',4), hidden=d.get('hidden',256), layers=d.get('layers',3))
        model.net.load_state_dict(d['state']); model.info = d['info']; return model
