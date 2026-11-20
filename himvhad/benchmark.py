"""Download pinned canonical features; never regenerate benchmark floats."""
import json,shutil,tarfile,urllib.request,zipfile
from pathlib import Path
from importlib.resources import files
import numpy as np
import pandas as pd
from .data import PHASES,boundaries
from .io import sha,write


def download(entry,path,local_repo=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if sha(path)!=entry['sha256']:raise ValueError('Existing file hash differs: '+str(path))
        return path
    part=path.with_suffix(path.suffix+'.part')
    if local_repo is not None:shutil.copyfile(Path(local_repo)/entry['path'],part)
    else:
        with urllib.request.urlopen(entry['url'],timeout=120) as inp,part.open('wb') as out:shutil.copyfileobj(inp,out)
    if part.stat().st_size!=entry['bytes'] or sha(part)!=entry['sha256']:raise ValueError('Downloaded artifact SHA256/size mismatch')
    part.replace(path);return path


def fetch_benchmark(root,datasets=('banksim','s_ffsd'),local_repo=None):
    root=Path(root);sources=json.loads(files('himvhad').joinpath('benchmark_sources.json').read_text());archive=download(sources['archive'],root/'.downloads/tabular.tar.gz',local_repo) if any(t!='cardsim' for t in datasets) else None
    for task in datasets:
        info=sources['datasets'][task];dest=root/task
        if (dest/'schema.json').exists():
            schema=json.loads((dest/'schema.json').read_text())
            for name,h in schema['prepared_files'].items():
                if sha(dest/name)!=h:raise ValueError('Prepared data changed: '+name)
            print('Verified existing prepared dataset '+task,flush=True);continue
        if dest.exists() and any(dest.iterdir()):raise FileExistsError('Use a fresh dataset folder: '+str(dest))
        dest.mkdir(parents=True,exist_ok=True)
        if task=='cardsim':
            ca=download(sources['cardsim_archive'],root/'.downloads/cardsim.tar.gz',local_repo)
            with tarfile.open(ca) as tar:
                members={m.name:m for m in tar.getmembers()}
                for key,entry in sources['cardsim_files'].items():
                    if not key.startswith('tabular/cardsim/'):continue
                    m=members[key]
                    if not m.isfile() or m.size!=entry['bytes']:raise ValueError('Invalid CardSim member')
                    target=dest/Path(key).name
                    with tar.extractfile(m) as inp,target.open('wb') as out:shutil.copyfileobj(inp,out)
                    if sha(target)!=entry['sha256']:raise ValueError('CardSim file hash mismatch')
            meta=json.loads((dest/'manifest.json').read_text());schema=dict(dataset=task,features=meta['features'],field_groups=info['field_groups'],view_columns=info['views'],window=4,splits=meta['splits'],canonical_source_commit=sources['commit'],protocol='Fixed independent CardSim world; whole-timestamp chronological60/20/20',prepared_files={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file()})
            write(dest/'schema.json',schema);print('Prepared sealed CardSim independent synthetic world',flush=True);continue
        with tarfile.open(archive) as tar:
            members={m.name:m for m in tar.getmembers()}
            for key,entry in sources['canonical_files'].items():
                if not key.startswith(task+'/'):continue
                m=members['tabular/'+key]
                if not m.isfile() or m.size!=entry['bytes']:raise ValueError('Invalid canonical archive member')
                target=dest/Path(key).name
                with tar.extractfile(m) as inp,target.open('wb') as out:shutil.copyfileobj(inp,out)
                if sha(target)!=entry['sha256']:raise ValueError('Canonical feature/target hash mismatch')
        raw=download(info['raw'],root/'.downloads'/info['raw']['name'],local_repo)
        with zipfile.ZipFile(raw) as z:
            with z.open(info['member']) as f:d=pd.read_csv(f,usecols=info['key_columns'])
        for col in info['key_columns']:
            if col!=info['time']:d[col]=d[col].astype(str).str.strip("'")
        times=d[info['time']].to_numpy();b=boundaries(times);split=np.searchsorted(b,times,side='left');meta=json.loads((dest/'manifest.json').read_text())
        for k,s in enumerate(PHASES):
            frame=d.loc[split==k]
            if len(frame)!=meta['splits'][s]['rows']:raise ValueError('Native-key alignment mismatch')
            order=np.argsort(frame[info['time']].to_numpy(),kind='stable')
            if not np.array_equal(order,np.arange(len(frame))):raise ValueError('Canonical row order must be chronological')
            np.savez_compressed(dest/f'{s}.keys.npz',**{f'k{i}':frame[col].to_numpy(dtype='U') for i,col in enumerate(info['views'])})
        schema=dict(dataset=task,features=meta['features'],field_groups=info['field_groups'],view_columns=info['views'],window=4,splits=meta['splits'],canonical_source_commit=sources['commit'],protocol=meta['protocol'],raw_sha256=sha(raw),canonical_manifest_sha256=sha(dest/'manifest.json'),prepared_files={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file()})
        write(dest/'schema.json',schema);print('Prepared exact canonical dataset '+task,flush=True)
    return root
