"""Label-free adaptation of the uploaded CNN -> compact KAN boundary pipeline.

The target is high training reconstruction error, NOT a fraud/correctness label.
Backbones stay frozen. Probe-space interventions are diagnostics, not proof of
financial causality. Linear B-spline KAN edges are an explicit implementation
choice, rather than the paper's unspecified pykan training configuration.
"""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn
from .hypergraph import NativeContext, assignments, masked


@dataclass(frozen=True)
class KnobConfig:
    probe_steps: int = 200
    student_steps: int = 150
    batch: int = 128
    channels: int = 8
    max_knobs: int = 10
    tolerance: float = .01
    fusion: float = .5
    low_quantile: float = .5
    high_quantile: float = .8
    validation_fraction: float = .2
    lr: float = .001
    spline_grid: int = 9

    def validate(self):
        if min(self.probe_steps, self.student_steps, self.batch, self.channels,
               self.max_knobs) < 1:
            raise ValueError('Positive training dimensions required')
        if self.batch < 2:
            raise ValueError('Balanced probe batches need at least2 rows')
        if not 0 < self.low_quantile < self.high_quantile < 1:
            raise ValueError('Ordered pseudo-target quantiles required')
        if not 0 < self.validation_fraction < .5 or not 0 <= self.fusion <= 1:
            raise ValueError('Invalid validation/fusion setting')
        if self.spline_grid < 3 or not 0 <= self.tolerance < 1:
            raise ValueError('Invalid spline grid/tolerance')


def hidden_states(model, x, views, rows=None):
    """Exact native clean states: projection, view-mean after each depth, fusion.

    Clean input is deliberate: the probe observes model state, not a second
    masked reconstruction detector. Nothing changes the parent's masked score.
    Only requested rows are retained; the graph context remains phase-local.
    """
    x = model._validate(x, views)
    rows = np.arange(len(x)) if rows is None else np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows) or (rows < 0).any() or (rows >= len(x)).any():
        raise ValueError('Distinct valid hidden-state rows required')
    ctx = NativeContext(x, views)
    model.net.eval()
    states = np.zeros((len(rows), model.layers + 2, model.hidden), dtype='f4')
    with torch.no_grad():
        initial = model.net.project(torch.as_tensor(ctx.enhanced(x, np.arange(len(x))), device=model.device))
        states[:, 0] = initial[torch.as_tensor(rows, device=model.device)].cpu().numpy()
        last = []
        for layers, edge in zip(model.net.branches, views):
            h = initial
            edge = torch.as_tensor(edge, device=model.device)
            for depth, layer in enumerate(layers):
                h = layer(h, edge)
                states[:, depth + 1] += h[torch.as_tensor(rows, device=model.device)].cpu().numpy() / model.nviews
            last.append(h[torch.as_tensor(rows, device=model.device)])
        net = model.net
        base = torch.softmax(net.gate(torch.relu(torch.cat(last, 1))), 1)
        shared = torch.tanh(net.mod_project(torch.stack(last).sum(0)))
        weights = (base + torch.cat([m(shared) for m in net.modulation], 1)).abs()
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        states[:, -1] = (torch.stack(last, 1) * weights[..., None]).sum(1).cpu().numpy()
    return states


def training_components(model, x, views, rows, batch=256):
    """Exact receptive-field masked errors for existing training targets only."""
    from .hypergraph import block_forward
    ctx = NativeContext(x, views)
    base = assignments(ctx.ids, model.seed + 991, len(model.groups))
    result = np.zeros((len(rows), len(model.groups)), dtype='f8')
    model.net.eval()
    with torch.no_grad():
        for turn in range(len(model.groups)):
            fields = (base + turn) % len(model.groups)
            value = masked(x, fields, model.groups)
            for start in range(0, len(rows), batch):
                target = rows[start:start + batch]
                plan = ctx.plan(target, layers=model.layers)
                needed = np.unique(np.concatenate([b[0][0] for b in plan]))
                pred = block_forward(model.net, ctx.enhanced(value, needed), plan,
                                     target, model.device, needed).cpu().numpy()
                error = (pred - x[target]) ** 2
                for g, cols in enumerate(model.groups):
                    mask = fields[target] == g
                    result[start + np.flatnonzero(mask), g] = error[mask][:, cols].mean(1)
    return result


class CNNBoundary(nn.Module):
    def __init__(self, width, channels=8):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.branches = nn.ModuleList([
            nn.Conv2d(1, channels, k, padding=(k[0]//2, k[1]//2))
            for k in [(9, 1), (1, 33), (5, 5)]])
        self.classifier = nn.Linear(channels * 3, 1)

    def maps(self, h):
        x = self.norm(h)[:, None]
        return [torch.relu(branch(x)) for branch in self.branches]

    def forward(self, h):
        return self.classifier(torch.cat([a.amax((2, 3)) for a in self.maps(h)], 1)).squeeze(1)


class SplineEdges(nn.Module):
    """Trainable univariate degree-1 B-spline edge functions + linear tails."""
    def __init__(self, inputs, outputs, grid=9):
        super().__init__()
        self.register_buffer('knots', torch.linspace(-4, 4, grid))
        self.coeff = nn.Parameter(torch.zeros(outputs, inputs, grid))
        self.linear = nn.Linear(inputs, outputs)

    def forward(self, x):
        spacing = self.knots[1] - self.knots[0]
        basis = torch.relu(1 - (x[..., None].clamp(-4, 4) - self.knots).abs() / spacing)
        return self.linear(x) + torch.einsum('nig,oig->no', basis, self.coeff)


class KANBoundary(nn.Module):
    def __init__(self, depth, indices, grid=9, attention=True, linear=False):
        super().__init__()
        self.register_buffer('indices', torch.as_tensor(indices, dtype=torch.long))
        self.layer_logits = nn.Parameter(torch.zeros(depth), requires_grad=attention)
        self.rule = nn.Linear(len(indices), 1) if linear else nn.Sequential(
            SplineEdges(len(indices), 2, grid), SplineEdges(2, 1, grid))

    def pool(self, h):
        return (h[:, :, self.indices] * self.layer_logits.softmax(0)[None, :, None]).sum(1)

    def forward(self, h):
        return self.rule(self.pool(h)).squeeze(1)


def pseudo_split(errors, seed, config):
    """Thresholds use the internal fitting portion, never the validation tail."""
    errors = np.asarray(errors, dtype='f8')
    if errors.ndim != 1 or len(errors) < 24 or not np.isfinite(errors).all():
        raise ValueError('At least24 finite training errors required')
    rng = np.random.default_rng(seed + 710)
    order = rng.permutation(len(errors)); cut = int(len(order)*(1-config.validation_fraction))
    fit, valid = order[:cut], order[cut:]
    lo, hi = np.quantile(errors[fit], [config.low_quantile, config.high_quantile])
    if not hi > lo:
        raise ValueError('Training errors cannot define distinct pseudo classes')
    labels = np.full(len(errors), -1, dtype='i8')
    labels[errors <= lo] = 0; labels[errors >= hi] = 1
    fit, valid = fit[labels[fit] >= 0], valid[labels[valid] >= 0]
    if any(len(np.unique(labels[i])) != 2 for i in [fit, valid]):
        raise ValueError('Both pseudo classes required in fitting and validation')
    return labels, fit, valid, dict(low=float(lo), high=float(hi))


def balanced_accuracy(labels, scores):
    return float(np.mean([np.mean((scores[labels == y] >= 0) == y) for y in [0, 1]]))


class KnobAdapter:
    def __init__(self, config=None, seed=42, device='cpu'):
        self.config = config or KnobConfig(); self.config.validate()
        self.seed, self.device = seed, device

    def _logits(self, model, h):
        model.eval(); result = []
        with torch.no_grad():
            for start in range(0, len(h), 512):
                result.append(model(torch.as_tensor(h[start:start+512], device=self.device)).cpu().numpy())
        return np.concatenate(result).astype('f8')

    def _fit(self, model, h, labels, fit, valid, steps, seed):
        torch.manual_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        rng = np.random.default_rng(seed)
        pools = [fit[labels[fit] == y] for y in [0, 1]]
        model.train()
        for _ in range(steps):
            ix = np.concatenate([rng.choice(p, self.config.batch//2, replace=True) for p in pools])
            opt.zero_grad(); score = model(torch.as_tensor(h[ix], device=self.device))
            loss = nn.functional.binary_cross_entropy_with_logits(score, torch.as_tensor(labels[ix], dtype=torch.float32, device=self.device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite boundary training loss')
            loss.backward(); opt.step()
        return balanced_accuracy(labels[valid], self._logits(model, h[valid]))

    def fit(self, states, errors):
        states = np.asarray(states, dtype='f4')
        if states.ndim != 3 or not np.isfinite(states).all() or len(states) != len(errors):
            raise ValueError('Finite NxLxD states and matching errors required')
        cfg = self.config
        labels, fit, valid, thresholds = pseudo_split(errors, self.seed, cfg)
        # All learned scaling, ranking and model fitting use internal fitting rows.
        self.center = states[fit].mean(0); self.scale = np.maximum(states[fit].std(0), 1e-4)
        h = np.clip((states - self.center) / self.scale, -8, 8).astype('f4')
        depth, width = states.shape[1:]
        torch.manual_seed(self.seed)
        self.teacher = CNNBoundary(width, cfg.channels).to(self.device)
        teacher_accuracy = self._fit(self.teacher, h, labels, fit, valid, cfg.probe_steps, self.seed)
        sums = np.zeros((2, depth, width)); counts = np.zeros(2)
        self.teacher.eval()
        with torch.no_grad():
            for start in range(0, len(fit), 128):
                ix = fit[start:start+128]
                # The paper permits a chosen branch; use the vertical branch.
                maps = self.teacher.maps(torch.as_tensor(h[ix], device=self.device))[0].mean(1).cpu().numpy()
                for y in [0, 1]:
                    sums[y] += maps[labels[ix] == y].sum(0); counts[y] += (labels[ix] == y).sum()
        self.importance = np.abs(sums[1]/counts[1] - sums[0]/counts[0]).sum(0)
        self.ranking = np.argsort(-self.importance, kind='stable')
        self.students = {}; sweep = []; selected = None
        for k in range(1, min(width, cfg.max_knobs)+1):
            torch.manual_seed(self.seed + 1000 + k)
            student = KANBoundary(depth, self.ranking[:k], cfg.spline_grid).to(self.device)
            accuracy = self._fit(student, h, labels, fit, valid, cfg.student_steps, self.seed+1000+k)
            sweep.append(dict(k=k, pseudo_validation_balanced_accuracy=accuracy))
            if accuracy >= teacher_accuracy - cfg.tolerance:
                selected = k; self.students['kan'] = student; break
            self.students['kan'] = student
        reached = selected is not None
        selected = selected or sweep[-1]['k']
        # Single random control fixed by RNG; no test/validation search over random sets.
        random_indices = np.random.default_rng(self.seed + 209).choice(width, selected, replace=False)
        controls = {}
        for name, indices, attention, linear in [
            ('random', random_indices, True, False), ('uniform', self.ranking[:selected], False, False),
            ('linear', self.ranking[:selected], True, True)]:
            torch.manual_seed(self.seed + 3000)
            model = KANBoundary(depth, indices, cfg.spline_grid, attention, linear).to(self.device)
            controls[name] = self._fit(model, h, labels, fit, valid, cfg.student_steps, self.seed+3000)
            self.students[name] = model
        # Explicit analytic surrogate, fitted to KAN logits, not real targets.
        self.symbolic_coef = np.linalg.lstsq(
            np.vstack([self._symbolic_features(h[fit]), np.sqrt(.001)*np.eye(1+4*selected)]),
            np.r_[self._logits(self.students['kan'], h[fit]), np.zeros(1+4*selected)], rcond=None)[0]
        target = self._logits(self.students['kan'], h[valid]); predicted = self._symbolic_features(h[valid]) @ self.symbolic_coef
        denom = float(np.sum((target-target.mean())**2))
        self.info = dict(config=asdict(cfg), seed=self.seed, actual_labels_read=False,
            proxy_target='high training masked reconstruction error, not fraud',
            teacher_validation_balanced_accuracy=teacher_accuracy, sweep=sweep,
            selected_k=selected, near_optimal_proxy_criterion_met=reached,
            selected_dimensions=self.ranking[:selected].tolist(), random_dimensions=random_indices.tolist(),
            controls_validation_balanced_accuracy=controls, pseudo_thresholds=thresholds,
            pseudo_fit_rows=len(fit), pseudo_validation_rows=len(valid),
            fit_rows=fit.tolist(), validation_rows=valid.tolist(),
            symbolic_validation_r2=1-float(np.sum((predicted-target)**2))/denom if denom>0 else None,
            symbolic_rule='c + sum_j [a_j*z_j + b_j*z_j^2 + c_j*tanh(z_j) + d_j*log(1+z_j^2)]',
            symbolic_coefficients=self.symbolic_coef.tolist(),
            layer_attention=self.students['kan'].layer_logits.softmax(0).detach().cpu().tolist())
        return self

    def _symbolic_features(self, h):
        with torch.no_grad():
            z = self.students['kan'].pool(torch.as_tensor(h, device=self.device)).cpu().numpy().astype('f8')
        return np.column_stack([np.ones(len(z)), z, z*z, np.tanh(z), np.log1p(z*z)])

    def logits(self, states):
        states = np.asarray(states, dtype='f4')
        if states.ndim != 3 or states.shape[1:] != self.center.shape or not len(states) or not np.isfinite(states).all():
            raise ValueError('Hidden-state schema mismatch')
        h = np.clip((states-self.center)/self.scale, -8, 8).astype('f4')
        result = dict(cnn=self._logits(self.teacher, h))
        result.update({name:self._logits(model, h) for name, model in self.students.items()})
        result['symbolic'] = self._symbolic_features(h) @ self.symbolic_coef
        return result

    def intervention_diagnostics(self, states):
        """Eq7-10 in the normalized probe-state space, no backbone patch claim."""
        h = torch.as_tensor(np.clip((states-self.center)/self.scale, -8, 8), device=self.device)
        model = self.students['kan']; model.eval()
        pooled = model.pool(h).detach().requires_grad_(True)
        value = model.rule(pooled).squeeze(1)
        grad = torch.autograd.grad(value.sum(), pooled)[0]
        norm = grad.norm(dim=1, keepdim=True); normal = grad/norm.clamp_min(1e-8)
        rng = np.random.default_rng(self.seed+910)
        random = torch.as_tensor(rng.normal(size=pooled.shape), dtype=pooled.dtype, device=self.device)
        random = random/random.norm(dim=1, keepdim=True).clamp_min(1e-8)
        tangent = random - (random*normal).sum(1, keepdim=True)*normal
        valid = (norm[:,0]>1e-7) & (tangent.norm(dim=1)>1e-7)
        tangent = tangent/tangent.norm(dim=1, keepdim=True).clamp_min(1e-8)
        alpha = model.layer_logits.softmax(0)
        records = []
        with torch.no_grad():
            for direction, vec in [('normal',normal), ('random',random), ('tangent',tangent)]:
                if direction == 'tangent' and pooled.shape[1] == 1:
                    records.append(dict(direction=direction,status='undefined_in_one_dimension'));continue
                for tau in [-.01, -.001, .001, .01]:
                    # Numerical safety cap, disclosed instead of huge OOD patches.
                    delta = (tau/norm.clamp_min(.05))*vec
                    patch = alpha[None,:,None]/alpha.square().sum()*delta[:,None,:]
                    identity = (patch*alpha[None,:,None]).sum(1)-delta
                    changed = model.rule(pooled+delta).squeeze(1)-value
                    keep = valid if direction=='tangent' else norm[:,0]>1e-7
                    records.append(dict(direction=direction,tau=tau,status='ok' if keep.any() else 'no_nonzero_gradient',
                        mean_logit_change=float(changed[keep].mean()) if keep.any() else None,
                        mean_absolute_logit_change=float(changed[keep].abs().mean()) if keep.any() else None,
                        positive_fraction=float((changed[keep]>0).float().mean()) if keep.any() else None,
                        pooled_patch_identity_max_abs=float(identity.abs().max())))
        return dict(scope='probe-space diagnostic; no evidence of real fraud causality or deployed correction',
                    gradient_floor=.05, rows=len(states), records=records)

    def save(self, path):
        torch.save(dict(format_version=1, config=asdict(self.config), seed=self.seed,
            center=torch.from_numpy(self.center), scale=torch.from_numpy(self.scale),
            importance=torch.from_numpy(self.importance), ranking=torch.from_numpy(self.ranking.copy()),
            teacher=self.teacher.state_dict(), students={k:v.state_dict() for k,v in self.students.items()},
            symbolic_coef=torch.from_numpy(self.symbolic_coef), info=self.info), path)

    @classmethod
    def load(cls, path, device='cpu'):
        data = torch.load(path, map_location='cpu', weights_only=True)
        obj = cls(KnobConfig(**data['config']), data['seed'], device)
        for k in ['center','scale','importance','ranking','symbolic_coef']:
            setattr(obj,k,data[k].numpy())
        obj.teacher = CNNBoundary(obj.center.shape[1],obj.config.channels).to(device)
        obj.teacher.load_state_dict(data['teacher']); obj.students = {}
        for name,state in data['students'].items():
            model = KANBoundary(obj.center.shape[0],state['indices'].numpy(),obj.config.spline_grid,
                                name!='uniform',name=='linear').to(device)
            model.load_state_dict(state);obj.students[name]=model
        obj.info = data['info'];return obj


class ScoreFusion:
    """Fit scalar proxy evidence on calibration only; fixed before real labels."""
    def __init__(self, weight=.5):
        if not 0<=weight<=1:raise ValueError('Fusion weight must be in [0,1]')
        self.weight=weight

    def fit(self, logits):
        self.sorted={k:np.sort(np.asarray(v,dtype='f8')) for k,v in logits.items()}
        if not self.sorted or any(v.ndim!=1 or len(v)<2 or not np.isfinite(v).all() for v in self.sorted.values()):
            raise ValueError('Finite calibration logits required')
        return self

    def scores(self, baseline, logits):
        baseline=np.asarray(baseline,dtype='f8')
        if set(logits)!=set(self.sorted):raise ValueError('Proxy schema mismatch')
        result={'baseline':baseline}
        for name,value in logits.items():
            value=np.asarray(value,dtype='f8');ref=self.sorted[name]
            if value.shape!=baseline.shape or not np.isfinite(value).all():raise ValueError('Invalid proxy logits')
            evidence=-np.log((1+len(ref)-np.searchsorted(ref,value,side='left'))/(len(ref)+1))
            result[name+'_only']=evidence
            result[name+'_fused']=(1-self.weight)*baseline+self.weight*evidence
        return result

    def save(self,path):np.savez_compressed(path,weight=self.weight,**self.sorted)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            obj=cls(float(z['weight']));obj.sorted={k:z[k] for k in z.files if k!='weight'}
        return obj
