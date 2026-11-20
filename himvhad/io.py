"""Atomic artifacts and explicit source/data provenance."""
import hashlib,json,os,platform,subprocess
from pathlib import Path
import numpy as np
import scipy,sklearn,torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix(path.suffix+'.tmp');part.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');part.replace(path)

def provenance():
    root=Path(__file__).resolve().parents[1];commit=None
    try:commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):pass
    return dict(git_commit=commit,source_sha256={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__)
