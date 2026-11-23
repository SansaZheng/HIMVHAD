"""Build paired tables directly from completed CQ artifacts, no model selection."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from himvhad.io import write,sha
from himvhad.knob_runner import VARIANTS


def flat(record):
    m=record['metrics'];r={k:record[k] for k in ['dataset','seed','variant']}
    for k,v in m.items():
        if k!='operating_points':r[k]=v
    for op,d in m['operating_points'].items():
        for k in ['precision','recall','f1','fpr','alarm_fraction','tp','fp','fn','tn']:
            r[op+'_'+k]=d[k] if d is not None else None
    return r


def csvwrite(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source,out=a.input,a.output;out.mkdir(parents=True,exist_ok=False)
    complete=json.loads((source/'complete.json').read_text());records=json.loads((source/'records.json').read_text());assert complete['status']=='complete' and len(records)==81
    rows=[flat(r) for r in records];write(out/'per_seed.json',records);csvwrite(out/'metrics.csv',rows)
    repo=Path(__file__).resolve().parents[1]
    old=json.loads((repo/'results/frozen/per_seed.json').read_text())
    old_skill=[r for r in old if r['variant']=='cm' and r['scoring']=='skill_tail']
    assert len(old_skill)==9
    csvwrite(out/'historical_skill_reference.csv',[flat(dict(r,variant='historical_cm_skill_tail')) for r in old_skill])
    summaries=[];paired=[];details=[]
    for d in ['banksim','s_ffsd','cardsim']:
        for v in VARIANTS:
            group=[r for r in rows if r['dataset']==d and r['variant']==v];assert len(group)==3
            summary=dict(dataset=d,variant=v,n=3)
            for key in ['average_precision','pr_auc_trapezoid','roc_auc','pot_precision','pot_recall','pot_f1','pot_fpr','pot_alarm_fraction','quantile_precision','quantile_recall','quantile_f1']:
                values=[r[key] for r in group if r[key] is not None]
                summary[key+'_mean']=float(np.mean(values)) if values else None;summary[key+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
                summary[key+'_available']=len(values)
            summaries.append(summary)
            for seed in [42,43,44]:
                base=next(r for r in rows if r['dataset']==d and r['seed']==seed and r['variant']=='baseline')
                row=next(r for r in group if r['seed']==seed)
                paired.append(dict(dataset=d,seed=seed,variant=v,**{k+'_delta_pp':100*(row[k]-base[k]) if row[k] is not None and base[k] is not None else None for k in ['average_precision','pr_auc_trapezoid','roc_auc','pot_f1','quantile_f1']}))
        for seed in [42,43,44]:
            folder=source/'runs'/d/str(seed);info=json.loads((folder/'adapter_info.json').read_text());stats=json.loads((folder/'scoring_complete.json').read_text())
            details.append(dict(dataset=d,seed=seed,selected_k=info['selected_k'],near_optimal=info['near_optimal_proxy_criterion_met'],teacher_proxy_ba=info['teacher_validation_balanced_accuracy'],student_proxy_ba=info['sweep'][-1]['pseudo_validation_balanced_accuracy'],symbolic_r2=info['symbolic_validation_r2'],seconds=stats['seconds'],peak_cuda_gb=stats['peak_cuda_allocated_gb']))
            dest=out/'runs'/d/str(seed);dest.mkdir(parents=True)
            for name in ['adapter_info.json','interventions.json','run_manifest.json','scoring_complete.json','label_access_gate.json','records.json']:
                shutil.copyfile(folder/name,dest/name)
            for variant in VARIANTS:
                f=dest/variant;f.mkdir()
                for name in ['thresholds.json','result.json']:shutil.copyfile(folder/variant/name,f/name)
            # Three fixed seed42 deployable adapters; all complete scores stay in CQ archive.
            if seed==42:
                for name in ['adapter.pt','baseline_calibrator.npz','fusion_calibrator.npz']:
                    shutil.copyfile(folder/name,dest/name)
    csvwrite(out/'summary.csv',summaries);csvwrite(out/'paired_differences.csv',paired);csvwrite(out/'proxy_diagnostics.csv',details)
    for name in ['complete.json','runtime.json','global_label_gate.json']:shutil.copyfile(source/name,out/name)
    def get(d,v):return next(r for r in summaries if r['dataset']==d and r['variant']==v)
    def fmt(row,key):
        m,sd,n=row[key+'_mean'],row[key+'_sd'],row[key+'_available']
        if m is None:return '失败/不可用（0/3）'
        text=f'{100*m:.2f} ± {100*sd:.2f}' if sd is not None else f'{100*m:.2f}'
        return text+(f'（n={n}/3）' if n<3 else '')
    lines=['# HIMVH 隐藏边界融合与消融实验（2026-11-23）','',
      '本报告由 `scripts/build_knob_report.py` 从完成的CQ逐种子记录生成。方法是用户所附论文机制的**零标签适配**，不是论文LLM成绩或原HIMVH监督成绩的复现。主方法固定为 `kan_fused`，不按各测试集最佳消融重新命名。',
      '', '[方法与复现命令](../../docs/mechanistic_boundary_CN.md) · [冻结协议](../../experiments/knob_protocol_20261123.json) · [逐种子指标](metrics.csv) · [配对差值](paired_differences.csv) · [代理诊断](proxy_diagnostics.csv)',
      '', '## 原版与融合版：三个骨干种子的配对对比','',
      '数值为百分数均值 ± 样本标准差；Δ为融合版减原版的百分点。AP为主排序指标；F1用预先固定的POT/q99规则，不搜索测试集最佳阈值。',
      '', '| 数据集 | 原版AP | 融合AP | ΔAP | 原版POT-F1 | 融合POT-F1 | 原版q99-F1 | 融合q99-F1 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    changes=[]
    for d in ['banksim','s_ffsd','cardsim']:
        b,k=get(d,'baseline'),get(d,'kan_fused');delta=100*(k['average_precision_mean']-b['average_precision_mean']);changes.append((d,delta))
        lines.append(f"| {d} | {fmt(b,'average_precision')} | {fmt(k,'average_precision')} | {delta:+.2f} | {fmt(b,'pot_f1')} | {fmt(k,'pot_f1')} | {fmt(b,'quantile_f1')} | {fmt(k,'quantile_f1')} |")
    lines+=['','仓库已有 `cm_skill_tail` 扩展也保留作参考（历史同检查点／划分结果，未冒充本轮新跑）：']
    for d in ['banksim','s_ffsd','cardsim']:
        refs=[r['metrics']['average_precision'] for r in old_skill if r['dataset']==d]
        new=get(d,'kan_fused')['average_precision_mean']
        lines.append(f"- {d}：原skill扩展AP {100*np.mean(refs):.2f} ± {100*np.std(refs,ddof=1):.2f}%；本轮主方法相对它 {100*(new-np.mean(refs)):+.2f}个百分点。")
    lines+=['','## 完整消融：平均精确率AP','', '| 设置 | BankSim | S-FFSD | CardSim |','|---|---:|---:|---:|']
    for v in VARIANTS:lines.append('| '+v+' | '+' | '.join(fmt(get(d,v),'average_precision') for d in ['banksim','s_ffsd','cardsim'])+' |')
    lines+=['','`baseline`逐点复用原cm_tail_flat，骨干参数不更新；`cnn_only/kan_only`去掉原版分数；`cnn_fused`不压缩到少量维度；`random_fused`替换关键维度；`uniform_fused`去掉层注意力；`linear_fused`去掉KAN非线性；`symbolic_fused`替换为解析近似。随机维度是固定单次控制，不是随机集合分布的全面检验。',
      '', '## 完整消融：POT-F1 / q99-F1','', '| 设置 | BankSim | S-FFSD | CardSim |','|---|---:|---:|---:|']
    for v in VARIANTS:lines.append('| '+v+' | '+' | '.join(fmt(get(d,v),'pot_f1')+' / '+fmt(get(d,v),'quantile_f1') for d in ['banksim','s_ffsd','cardsim'])+' |')
    lines+=['','完整Precision、Recall、PR-AUC、ROC-AUC、FPR、报警比例和混淆矩阵在 `metrics.csv/per_seed.json`，各数值阈值及POT状态保存在 `runs/<dataset>/<seed>/<variant>/thresholds.json`。三个种子不足以宣称统计显著。',
      '', '## 哪些模块实际有帮助','']
    for d in ['banksim','s_ffsd','cardsim']:
        b,k=get(d,'baseline'),get(d,'kan_fused');best=max((get(d,v) for v in VARIANTS),key=lambda r:r['average_precision_mean'])
        dif=100*(k['average_precision_mean']-b['average_precision_mean']);cnn=100*(k['average_precision_mean']-get(d,'cnn_fused')['average_precision_mean']);random=100*(k['average_precision_mean']-get(d,'random_fused')['average_precision_mean']);linear=100*(k['average_precision_mean']-get(d,'linear_fused')['average_precision_mean'])
        lines.append(f"- **{d}**：主方法ΔAP {dif:+.2f}个百分点；相对全维CNN融合 {cnn:+.2f}，相对同k随机维度 {random:+.2f}，相对线性边界 {linear:+.2f}。本次所列消融中AP最高为 `{best['variant']}`（{fmt(best,'average_precision')}%），它是事后描述，不替代预先指定主方法。")
    lines+=['','## 紧凑性与解析近似是否成立','', '| 数据集 | 种子 | 维度k | 达到代理近最优判据 | CNN代理BA | KAN代理BA | 解析式验证R² |','|---|---:|---:|---|---:|---:|---:|']
    for r in details:
        r2=f"{r['symbolic_r2']:.4f}" if r['symbolic_r2'] is not None else '不可用（常数目标）'
        lines.append(f"| {r['dataset']} | {r['seed']} | {r['selected_k']} | {'是' if r['near_optimal'] else '否，达到搜索上限'} | {r['teacher_proxy_ba']:.4f} | {r['student_proxy_ba']:.4f} | {r2} |")
    lines+=['','**这些BA只是识别高／低训练重构误差的能力，不能与欺诈检测准确率混用。** 未达到近最优判据时不声称发现了近无损小子空间；低R²或负R²表示解析规则不能忠实近似KAN，仍保留其检测消融。各组维度、注意力、公式系数和完整k搜索记录可查 `adapter_info.json`。',
      '', '## 如何解释提升与下降','',
      '以下是结合任务差异和消融的解释，属于机制推断，不是已证实的金融因果结论：',
      '', '- 代理目标只反映模型重构困难，可能集中在罕见正常交易、字段尺度或关系稀疏性。真实欺诈可以被模型重构得很好；因此高代理BA并不保证高欺诈AP。',
      '- 冻结骨干没有因新模块学到新的反欺诈信息。新模块主要重新组织已有状态；如果CNN单独或KAN单独明显弱于基线，融合只会把噪声加入原分数。',
      '- 若完整KAN不优于同k随机维度或线性控制，本次证据不支持“选出的关键维度／KAN非线性是收益来源”。少量图网络深度也不同于原文深层语言模型，不能借用其机制结论。',
      '- 原版的字段校准有时已携带较强信息。统一0.5融合可能稀释它；权重固定是为了避免用本轮测试标签补救结果。',
      '- AP与固定阈值F1可能相反：AP评价全部排序，POT/q99只看一个校准工作点；校准与测试分布变化会改变报警率。不能只保留其中改善的指标。',
      '- 解析式的R²低表示蒸馏损失；探针法向能改变自身logit是局部数学性质，不代表改善真实欺诈识别。',
      '', '## 公平性与证据边界','',
      '三个数据集原始划分、目标行、测试样本、训练缩放器和骨干检查点不变。新增模块只用训练特征和自身重构误差；内部验证只在原训练分区。校准用于分数尾部与阈值。所有9组评分及阈值在本轮第一次打开测试标签前冻结；`global_label_gate.json`及各组记录提供SHA链。基线误差直接取已核验历史文件，防止跨CPU/CUDA重算微差混入方法改进。',
      '', 'BankSim和S-FFSD是回顾性测试；CardSim是此前生成的单个合成世界，本次并未新增独立真实数据。附加探针、k搜索和对照有额外计算成本。方向干预只在内部验证的标准化探针状态做诊断，未声称对完整骨干做因果干预。',
      '', f"完成{complete['groups']}组冻结骨干、{complete['configurations']}个评分配置；新增骨干训练0次。完整实验耗时{complete['seconds']/60:.2f}分钟，单进程峰值RAM{complete['peak_ram_linux_gb']:.2f}GiB，峰值CUDA分配{complete['peak_cuda_allocated_gb']:.2f}GiB。运行环境与精确代码提交在 `runtime.json`。时长不含CQ排队、环境作业与GPU检查。", 
      '', '## 产物与重新运行','',
      '本目录发布全部逐种子指标、差值、阈值、代理搜索记录和SHA。固定seed42的三个适配器及校准器也保留在 `runs/<dataset>/42/`，不是按最佳种子挑选。全部逐点分数、训练代理误差及其余种子模型保留在本轮CQ完整结果归档；其SHA在标签访问记录中，重现命令见方法说明。',
      '', 'English: Nine paired frozen-backbone runs evaluate a CNN/KAN hidden-boundary adaptation with training reconstruction-error proxy targets. The unchanged baseline, all ablations and negative outcomes are retained. Proxy compactness and directional sensitivity do not demonstrate fraud correctness, statistical significance or a causal financial mechanism.', '']
    (out/'README_CN.md').write_text('\n'.join(lines))
    write(out/'artifact_manifest.json',dict(source_complete_sha256=sha(source/'complete.json'),source_records_sha256=sha(source/'records.json'),source_global_gate_sha256=sha(source/'global_label_gate.json'),groups=9,configurations=81,files_sha256={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))
    print(json.dumps(dict(primary_delta_AP_pp=dict(changes),output=str(out)),ensure_ascii=False))


if __name__=='__main__':main()
