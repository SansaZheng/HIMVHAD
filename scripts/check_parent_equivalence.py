"""Optional exact training parity against the pinned source repository."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from himvhad.model import HIMVHAnomalyDetector
from himvhad.network import hyperedges
from himvhad.io import sha,write,provenance
p=argparse.ArgumentParser();p.add_argument('--parent-repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();manifest=json.loads(Path('PORTING_MANIFEST.json').read_text())
for name,h in manifest['files'].items():assert sha(a.parent_repo/name)==h,name
sys.path.insert(0,str(a.parent_repo));from finad_repair.himvh import HIMVHRepair
started=time.monotonic();torch.set_num_threads(1);x=np.random.default_rng(19).normal(size=(24,61)).astype('f4');views=[hyperedges(np.arange(24)//6),hyperedges(np.arange(24)%4)]*2;checked=[]
for native,consistent in [(False,False),(True,False),(False,True),(True,True)]:
 old=HIMVHRepair(61,'banksim',42,'cpu',native,consistent,steps=2).fit(x,views,np.arange(len(x)));new=HIMVHAnomalyDetector(61,'banksim',42,'cpu',native,consistent,steps=2).fit(x,views,np.arange(len(x)))
 for name,value in old.net.state_dict().items():assert torch.equal(value,new.net.state_dict()[name]),name
 np.testing.assert_array_equal(old.components(x,views),new.components(x,views));checked.append(dict(native=native,consistent=consistent,state_tensors_bitwise_equal=True,components_bitwise_equal=True))
write(a.output,dict(status='passed',source=provenance(),parent_commit=manifest['parent_commit'],toy_training_only=True,new_financial_fits=0,checks=checked,seconds=time.monotonic()-started));print('Exact parent training/state/component equivalence passed4/4 controls')
