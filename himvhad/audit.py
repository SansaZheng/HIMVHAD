"""Label-independent checkpoint probes using exact sparse receptive fields."""
import numpy as np
import torch
from .hypergraph import NativeContext,block_forward,assignments,masked

def probe_rows(n):return np.unique(np.r_[np.arange(min(16,n)),np.linspace(0,n-1,min(17,n),dtype=np.int64)])

def checkpoint_components(model,x,views,rows):
    ctx=NativeContext(x,views);plan=ctx.plan(rows,layers=model.layers);needed=np.unique(np.concatenate([b[0][0] for b in plan]));base=assignments(ctx.ids,model.seed+991,len(model.groups));result=np.zeros((len(rows),len(model.groups)));model.net.eval()
    with torch.no_grad():
        for turn in range(len(model.groups) if model.consistent else 1):
            fields=(base+turn)%len(model.groups);value=masked(x,fields,model.groups) if model.consistent else x;enhanced=ctx.enhanced(value,needed);pred=block_forward(model.net,enhanced,plan,rows,'cpu',needed).numpy();error=(pred-x[rows])**2
            for g,cols in enumerate(model.groups):
                selected=fields[rows]==g if model.consistent else np.ones(len(rows),bool);result[selected,g]=error[selected][:,cols].mean(1)
    return result
