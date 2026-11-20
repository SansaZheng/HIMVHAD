"""Import previously published HIM-only tables without relabeling them as new runs."""
import argparse,csv,json
from pathlib import Path
from himvhad.io import sha,write
p=argparse.ArgumentParser();p.add_argument('--parent-repo',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('results/history'));a=p.parse_args();manifest=[]
for name in ['recent_repair_20261117','roleweight_20261117']:
 source=a.parent_repo/'results'/name;out=a.output/name;out.mkdir(parents=True,exist_ok=True)
 rec=[r for r in json.loads((source/'per_seed.json').read_text()) if r['family']=='himvh'];write(out/'per_seed.json',rec)
 for n in ['summary.csv','metrics.csv','paired_effects.csv']:
  with (source/n).open() as f:reader=csv.DictReader(f);keys=reader.fieldnames;rows=[r for r in reader if r['family']=='himvh']
  with (out/n).open('w',newline='') as f:writer=csv.DictWriter(f,keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)
 proof=json.loads((source/'verification.json').read_text());assert proof['status']=='passed';record=dict(source_repository='SansaZheng/Financial_unsuperviesdAD',source_commit='f0a6c7fa00eb54fba0f76b421cc11f66b9b536fe',source_path='results/'+name,scope='imported historical HIM evidence; not newly fitted in HIMVHAD',configurations=len(rec),source_files_sha256={n:sha(source/n) for n in ['per_seed.json','summary.csv','metrics.csv','paired_effects.csv','verification.json']},upstream_verification_status='passed');write(out/'provenance.json',record);manifest.append(record)
write(a.output/'index.json',manifest)
(a.output/'README_CN.md').write_text('''# 导入的历史HIMVH证据

本目录从父项目固定提交抽取HIMVH相关记录：修正轮42配置／84操作点，字段评分轮96配置／192操作点。包含重用同一骨干、重复评分对照，不能将这些配置数相加视作独立训练次数。来源文件SHA、路径与原核验状态在各provenance.json，数值没有重新拟合或筛选。

本次独立包重新计算的144评分配置、18冻结模型核验与CardSim结果见[当前报告](../frozen/README_CN.md)。历史记录与新重评分分别保存，不冒称本次从头训练。
''')
print('Historical HIM evidence imported with provenance')
