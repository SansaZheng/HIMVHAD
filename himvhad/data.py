"""Phase-local native views and label-blind temporal feature preparation."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from .network import hyperedges
from .io import write,sha
PHASES=('train','calibration','test')

def boundaries(times):
    counts=pd.Series(times).value_counts().sort_index();c=counts.cumsum().to_numpy()
    return [counts.index[min(np.searchsorted(c,c[-1]*q),len(counts)-1)] for q in [.6,.8]]

def signed_log(x):
    x=np.asarray(x,dtype='f8');return np.sign(x)*np.log1p(np.abs(x))

def hash_onehot(values,bins):
    s=pd.Series(values).fillna('<missing>').astype(str);mapping={v:int.from_bytes(hashlib.sha256(v.encode()).digest()[:4],'little')%bins for v in s.unique()}
    return np.eye(bins,dtype='f4')[s.map(mapping).to_numpy()]

def load_data(path):
    path=Path(path);schema=json.loads((path/'schema.json').read_text());width=len(schema['features']);x={s:np.memmap(path/f'{s}.x.f32',mode='r',dtype='f4',shape=(schema['splits'][s]['rows'],width)) for s in PHASES}
    return schema,x

def load_views(path,phase,indices=None):
    path=Path(path)
    schema=json.loads((path/'schema.json').read_text())
    with np.load(path/f'{phase}.keys.npz',allow_pickle=False) as z:keys=[z[k] for k in sorted(z.files,key=lambda k:int(k[1:]))]
    if schema.get('view_encoding')=='explicit_edges':
        n=schema['splits'][phase]['rows']
        if len(keys)!=len(schema['view_columns']) or any(
            e.ndim!=2 or e.shape[1]<2 or e.dtype.kind not in 'iu' or (e<0).any() or (e>=n).any()
            for e in keys
        ):
            raise ValueError('Invalid phase-local explicit hyperedges')
        if indices is None:
            return keys
        local=np.full(n,-1,dtype=np.int64)
        local[indices]=np.arange(len(indices))
        remapped=[local[edges] for edges in keys]
        return [edges[(edges>=0).all(1)] for edges in remapped]
    return [hyperedges(k if indices is None else k[indices],4) for k in keys]

def prepare_csv(csv_path,config_path,output):
    config=json.loads(Path(config_path).read_text());output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh prepared-data directory')
    if len(config['views'])<2:raise ValueError('Need at least two native view keys')
    if config.get('label') in config.get('numeric',[]) or config.get('label') in config.get('categorical',{}):raise ValueError('Label cannot be an input feature')
    if config.get('label') in config['views']:raise ValueError('Label cannot be a view key')
    d=pd.read_csv(csv_path);time=config['time']
    if d[time].isna().any():raise ValueError('Missing timestamps')
    # Numeric clocks are ordered numerically; strings must parse as datetimes.
    t=d[time] if pd.api.types.is_numeric_dtype(d[time]) else pd.to_datetime(d[time],utc=True,errors='raise').astype('int64')
    order=np.argsort(t.to_numpy(),kind='stable');d=d.iloc[order].reset_index(drop=True);t=np.asarray(t)[order];b=boundaries(t);split=np.searchsorted(b,t,side='left')
    values=[];names=[];groups=[]
    for col in config.get('numeric',[]):
        v=pd.to_numeric(d[col],errors='raise').to_numpy();v=signed_log(v) if col in config.get('signed_log',[]) else v;groups.append([len(names)]);names.append(col);values.append(v[:,None])
    for col,bins in config.get('categorical',{}).items():
        if int(bins)<2:raise ValueError('Need >=2 hash bins')
        start=len(names);values.append(hash_onehot(d[col],int(bins)));names += [col+'_hash_'+str(i) for i in range(int(bins))];groups.append(list(range(start,len(names))))
    if not values:raise ValueError('No feature columns')
    x=np.column_stack(values).astype('f4')
    if not np.isfinite(x).all():raise ValueError('Numeric features must be finite; preprocess missing values explicitly')
    if any(not np.any(split==k) for k in range(3)):raise ValueError('Need three nonempty chronological partitions')
    # Data preparation may write evaluation labels; model training never opens them.
    y=None
    if config.get('label'):
        labels=d[config['label']];mapping=config.get('label_map');y=labels.astype(str).map(mapping).to_numpy() if mapping else labels.to_numpy()
        if not np.isin(y,[-1,0,1]).all():raise ValueError('Labels must map to0/1/-1; use label_map for unknowns')
    output.mkdir(parents=True);schema=dict(dataset=config.get('dataset','custom'),features=names,field_groups=groups,view_columns=config['views'],window=4,protocol='whole-timestamp chronological60/20/20; offline phase-local views',time_boundaries=list(map(str,b)),splits={},raw_sha256=sha(csv_path))
    for k,s in enumerate(PHASES):
        mask=split==k;x[mask].tofile(output/f'{s}.x.f32');schema['splits'][s]=dict(rows=int(mask.sum()));np.savez_compressed(output/f'{s}.keys.npz',**{f'k{i}':d.loc[mask,col].astype(str).to_numpy(dtype='U') for i,col in enumerate(config['views'])})
        if s=='test' and y is not None:np.asarray(y[mask],dtype='i1').tofile(output/'test.y.i8')
    write(output/'schema.json',schema);return schema

def demo_data(output,seed=31415):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(seed);schema=dict(dataset='synthetic_demo_not_financial_evidence',features=[f'x{i}' for i in range(6)],field_groups=[[0],[1,2],[3,4,5]],view_columns=['demo_sender','demo_target','demo_location','demo_type'],window=4,splits={})
    for s,n in [('train',256),('calibration',128),('test',128)]:
        x=rng.normal(size=(n,6)).astype('f4');y=np.zeros(n,dtype='i1')
        if s=='test':y[-8:]=1;x[-8:,0]+=5;y.tofile(output/'test.y.i8')
        x.tofile(output/f'{s}.x.f32');schema['splits'][s]=dict(rows=n);np.savez_compressed(output/f'{s}.keys.npz',**{f'k{i}':((np.arange(n)//(i+2))%8).astype('U') for i in range(4)})
    write(output/'schema.json',schema);return schema
