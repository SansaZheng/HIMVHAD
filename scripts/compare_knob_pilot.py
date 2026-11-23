"""Retain the initial RNG-confounded pilot without mixing it into final means."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
from himvhad.io import write,sha
from himvhad.knob_runner import VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--final',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    pilot=json.loads((a.pilot/'records.json').read_text());final=json.loads((a.final/'records.json').read_text());assert len(pilot)==len(final)==81
    out=a.report/'pilot_rng';out.mkdir(exist_ok=False);write(out/'per_seed.json',pilot)
    for name in ['runtime.json','complete.json','global_label_gate.json']:shutil.copyfile(a.pilot/name,out/name)
    rows=[];parity=[]
    for dataset in ['banksim','s_ffsd','cardsim']:
        for variant in VARIANTS:
            first=[r for r in pilot if r['dataset']==dataset and r['variant']==variant];last=[r for r in final if r['dataset']==dataset and r['variant']==variant]
            before=np.array([r['metrics']['average_precision'] for r in first]);after=np.array([r['metrics']['average_precision'] for r in last]);rows.append(dict(dataset=dataset,variant=variant,pilot_AP_mean=float(before.mean()),paired_AP_mean=float(after.mean()),rerun_minus_pilot_AP_pp=float(100*(after.mean()-before.mean()))))
        for seed in [42,43,44]:
            for variant in ['baseline','cnn_only','cnn_fused','kan_only','kan_fused','symbolic_fused']:
                r1=a.pilot/'runs'/dataset/str(seed)/variant/'scores.npz';r2=a.final/'runs'/dataset/str(seed)/variant/'scores.npz'
                with np.load(r1) as x,np.load(r2) as y:
                    parity.append(dict(dataset=dataset,seed=seed,variant=variant,calibration_bitwise_equal=bool(np.array_equal(x['calibration'],y['calibration'])),test_bitwise_equal=bool(np.array_equal(x['test'],y['test'])),test_max_abs=float(np.max(np.abs(x['test']-y['test'])))))
    for name,values in [('summary.csv',rows),('unchanged_path_parity.csv',parity)]:
        with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator="\n");w.writeheader();w.writerows(values)
    exact=sum(r['calibration_bitwise_equal'] and r['test_bitwise_equal'] for r in parity)
    write(out/'provenance.json',dict(scope='RNG-confounded pilot;not strict component ablation;not combined with final results',pilot_records_sha256=sha(a.pilot/'records.json'),final_records_sha256=sha(a.final/'records.json'),unchanged_formula_paths=len(parity),bitwise_equal_paths=exact,model_fits='Same9 frozen backbones reused in two adapter runs;no new backbone fitting'))
    lines=['','## 初跑保留与随机流修正','',
      '初跑j-1123-21f2ed的完整指标保存在[pilot记录](pilot_rng/per_seed.json)。静态复核发现，初版的三个模块对照用了不同初始化和小批次流；修正版统一为选定KAN的随机流，新增测试核验同形参数初值逐项相等。修正发生在查看任何新金融测试指标之前，未根据结果选参数。',
      '', '初跑没有删去，也不纳入主表均值。下表仅说明实现修正的影响，不能把初跑当作额外独立种子或新数据。',
      '', '| 数据集 | 对照 | 初跑AP | 对齐随机流后AP | 差值/百分点 |','|---|---|---:|---:|---:|']
    for r in rows:
        if r['variant'] in ['random_fused','uniform_fused','linear_fused']:
            lines.append(f"| {r['dataset']} | {r['variant']} | {100*r['pilot_AP_mean']:.2f} | {100*r['paired_AP_mean']:.2f} | {r['rerun_minus_pilot_AP_pp']:+.2f} |")
    lines+=['',f'原版、CNN、主KAN和解析近似路径的公式未改变；两个独立运行中，{exact}/{len(parity)}条路径的校准与测试分数按位相同。其余差异（若存在）在[逐路径检查](pilot_rng/unchanged_path_parity.csv)中披露，不冒充随机流修正带来的方法增益。','']
    with (a.report/'README_CN.md').open('a') as f:f.write('\n'.join(lines))
    print(f'Pilot retained separately;unchanged-formula parity {exact}/{len(parity)}')


if __name__=='__main__':main()
