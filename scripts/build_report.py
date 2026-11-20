"""Publish audited frozen-model results and clearly separated historical evidence."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
from himvhad.io import sha,write
METRICS=['average_precision','pr_auc_trapezoid','roc_auc','precision','recall','f1','fpr']
NAMES=dict(field='字段均值',flat='按维数',tail='字段尾证据等权',tail_flat='尾证据＋维数',skill_tail='重构增益＋尾证据（原预登记主）',skill_raw='重构增益＋原误差',skill_equal_tail='重构增益无维数先验',skill_hard_tail='重构增益无权重下限')
TASKS=['banksim','s_ffsd','cardsim']

def csvsave(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)

def readcsv(path):
    with path.open() as f:return list(csv.DictReader(f))

def fmt(v):return 'N/A' if v is None else f'{100*v:.2f}'
def ms(r,m):return 'N/A' if r[m+'_mean'] is None else fmt(r[m+'_mean'])+' ± '+fmt(r[m+'_std'])

def summarize(records):
    rows=[]
    for r in records:
        for op,value in r['metrics']['operating_points'].items():
            item={k:r[k] for k in ['dataset','variant','seed','scoring']};item.update(operating_point=op,status='ok' if value is not None else 'failed');item.update({k:v for k,v in r['metrics'].items() if k!='operating_points'});item.update({k:value[k] if value else None for k in ['precision','recall','f1','fpr','tp','fp','fn','tn','alarm_fraction']});rows.append(item)
    summary=[]
    for task,base,method,op in sorted({(r['dataset'],r['variant'],r['scoring'],r['operating_point']) for r in rows}):
        g=[r for r in rows if (r['dataset'],r['variant'],r['scoring'],r['operating_point'])==(task,base,method,op)];assert len(g)==3;item=dict(dataset=task,variant=base,scoring=method,operating_point=op,seeds=3,threshold_successes=sum(r['status']=='ok' for r in g))
        for m in METRICS:
            v=[r[m] for r in g if r[m] is not None];item[m+'_mean']=float(np.mean(v)) if len(v)==3 else None;item[m+'_std']=float(np.std(v,ddof=1)) if len(v)==3 else None
        for m in ['evaluated','positives','normal','unknown_excluded','prevalence']:assert len({r[m] for r in g})==1;item[m]=g[0][m]
        summary.append(item)
    return rows,summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--verified',type=Path,required=True);p.add_argument('--parent-repo',type=Path,required=True);p.add_argument('--old-parents',type=Path,required=True);p.add_argument('--new-results',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('results/frozen'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    proof=json.loads((a.verified/'verification.json').read_text());assert proof['status']=='passed' and proof['evaluated_configurations']==144
    for name in ['per_seed.json','verification.json','parent_models.json','weights.json','baseline_references.json']:shutil.copyfile(a.verified/name,a.output/name)
    rec=json.loads((a.verified/'per_seed.json').read_text());rows,summary=summarize(rec);csvsave(a.output/'metrics.csv',rows);csvsave(a.output/'summary.csv',summary);assert len(rows)==288 and len(summary)==96
    sx={(r['dataset'],r['variant'],r['scoring'],r['operating_point']):r for r in summary};effects=[];ix={(r['dataset'],r['variant'],r['scoring'],r['seed'],r['operating_point']):r for r in rows}
    for d in TASKS:
        for b in ['m','cm']:
            for new,old in [('tail_flat','flat'),('skill_tail','tail_flat'),('skill_equal_tail','skill_tail'),('skill_hard_tail','skill_tail'),('skill_tail','skill_raw'),('skill_tail','field')]:
                for op in ['pot','quantile']:
                    row=dict(dataset=d,variant=b,new=new,old=old,operating_point=op)
                    for metric in METRICS:
                        vals=[ix[d,b,new,s,op][metric]-ix[d,b,old,s,op][metric] for s in [42,43,44] if ix[d,b,new,s,op][metric] is not None and ix[d,b,old,s,op][metric] is not None];row[metric+'_delta_mean']=float(np.mean(vals)) if len(vals)==3 else None;row[metric+'_delta_std']=float(np.std(vals,ddof=1)) if len(vals)==3 else None
                    effects.append(row)
    csvsave(a.output/'paired_effects.csv',effects)
    # Baselines are historical references; this repository does not train them.
    baselines=[]
    for r in readcsv(a.parent_repo/'results/roleweight_20261117/combined_transaction_summary.csv'):
        if r['dataset'] not in ['banksim','s_ffsd'] or r['method'] not in ['hbos','lof','vae']:continue
        q={k:r[k] for k in ['dataset','method','operating_point']};q.update({k:float(v) if v else None for k,v in r.items() if k.endswith(('_mean','_std'))});q['source_kind']='imported_verified_reference';baselines.append(q)
    refs=json.loads((a.verified/'baseline_references.json').read_text())
    for method in ['hbos','lof','vae']:
        group=[r for r in refs if r['variant']==method]
        for op in ['pot','quantile']:
            row=dict(dataset='cardsim',method=method,operating_point=op,source_kind='historical_CQ_scores_reaudited')
            for metric in METRICS:
                v=[r['metrics'][metric] if metric in r['metrics'] else (r['metrics']['operating_points'][op][metric] if r['metrics']['operating_points'][op] else None) for r in group];row[metric+'_mean']=float(np.mean(v)) if all(x is not None for x in v) else None;row[metric+'_std']=float(np.std(v,ddof=1)) if all(x is not None for x in v) else None
            baselines.append(row)
    assert len(baselines)==18;csvsave(a.output/'baseline_reference_summary.csv',baselines)
    old=json.loads((a.parent_repo/'results/roleweight_20261117/per_seed.json').read_text());checks=[]
    for r in rec:
        if r['dataset']=='cardsim' or r['scoring'].startswith('skill_'):continue
        ref=next(z for z in old if (z['family'],z['dataset'],z['variant'],z['seed'])==('himvh',r['dataset'],r['variant']+'_'+r['scoring'],r['seed']));assert r['metrics']==ref['metrics'];checks.append(dict(dataset=r['dataset'],variant=r['variant'],scoring=r['scoring'],seed=r['seed'],metrics_exact=True))
    assert len(checks)==48;write(a.output/'historical_metric_identity.json',dict(status='passed',matching_configurations=48,checks=checks))
    # Portable small examples: fixed cm/42, never selected as best seeds.
    pretrained=[]
    for task in TASKS:
        source=(a.new_results if task=='cardsim' else a.old_parents)/'tabular'/task/'himvh/cm/42';scores=a.verified/'scores'/task/'cm/42';target=Path('examples/pretrained')/task;target.mkdir(parents=True,exist_ok=True)
        for name in ['model.pt','preprocessor.npz']:shutil.copyfile(source/name,target/name)
        for name in ['field_calibrator.npz','skill_calibrator.npz']:shutil.copyfile(scores/name,target/name)
        schema=json.loads((a.data/task/'schema.json').read_text());write(target/'input_schema.json',dict(features=schema['features'],field_groups=schema['field_groups']))
        for name in NAMES:
            (target/name).mkdir(exist_ok=True);shutil.copyfile(scores/name/'thresholds.json',target/name/'thresholds.json')
        info=dict(dataset=task,variant='cm',seed=42,origin='copied_frozen_checkpoint_not_new_training',files_sha256={str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file() and p.name!='provenance.json'});write(target/'provenance.json',info);pretrained.append(info)
    write(a.output/'pretrained_examples.json',pretrained)
    t=['# HIMVH-Zero 冻结模型复现与评分结果','','日期：2026-11-21。**本次使用18个已训练冻结模型重新计算144组评分配置／288条操作点，新金融神经网络训练次数为0。** 这验证独立包与历史模型兼容及评分复现，不冒充重新训练。模型来源：BankSim／S-FFSD的12个旧m/cm检查点，加上CQ任务j-1117-8c114e已完成的6个CardSim检查点。','','原方法训练是有标签分类；本项目为零标签重构适配，不能用下面的分数复刻论文监督成绩。','','## 主结果与已有评分对照','','下表使用原预登记的cm-skill-tail作为固定扩展主方案，不按数据集改选最高消融。百分数，3种子均值±样本标准差。','','| 数据集 | cm-field AP | cm-tail-flat AP | cm-skill-tail AP | skill相对tail-flat AP差值（百分点） | skill POT F1 | skill q99 F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for d in TASKS:
        f=sx[d,'cm','field','pot'];b=sx[d,'cm','tail_flat','pot'];n=sx[d,'cm','skill_tail','pot'];q=sx[d,'cm','skill_tail','quantile'];t.append('| '+d+' | '+' | '.join([ms(f,'average_precision'),ms(b,'average_precision'),ms(n,'average_precision'),f'{100*(n["average_precision_mean"]-b["average_precision_mean"]):+.2f}',ms(n,'f1'),ms(q,'f1')])+' |')
    t+=['','BankSim／S-FFSD已被用于既有多轮分析，属于回顾性开发。CardSim模拟参数及skill公式在生成其数据前固定；这是一个独立合成世界的协议检验，**不是真实银行外部验证，也不是把旧模型直接零样本迁移**。新世界各自的训练／校准／测试分区按时间60/20/20，所有80107笔交易保留。','', '![冻结模型评分比较](comparison.png)', '', '## 全部评分和上下文消融']
    for d in TASKS:
        for b in ['m','cm']:
            t+=['','### '+d+'／'+b,'','| 评分 | AP ↑ | ROC ↑ | POT F1 ↑ | q99 F1 ↑ | POT FPR ↓ | q99 FPR ↓ |','|---|---:|---:|---:|---:|---:|---:|']
            for name in NAMES:
                r=sx[d,b,name,'pot'];q=sx[d,b,name,'quantile'];t.append('| '+NAMES[name]+' | '+' | '.join([ms(r,'average_precision'),ms(r,'roc_auc'),ms(r,'f1'),ms(q,'f1'),ms(r,'fpr'),ms(q,'fpr')])+' |')
    t+=['','## 能说明什么，不能说明什么']
    for d in TASKS:
        n=sx[d,'cm','skill_tail','pot'];f=sx[d,'cm','field','pot'];b=sx[d,'cm','tail_flat','pot'];delta=n['average_precision_mean']-b['average_precision_mean'];text=f'**{d}：** 相对字段均值AP变化{100*(n["average_precision_mean"]-f["average_precision_mean"]):+.2f}个百分点，但相对已有维数加权尾证据仅变化{100*delta:+.2f}。'
        text+=('本数据上的skill权重没有超过该对照，不能将尺度或维数先验收益归功于新的重构增益。' if delta<=0 else '本数据上存在超过同尺度宽度先验的收益，仍需结合种子差异及FPR，不自动构成显著性或广泛泛化结论。');t+=['',text]
    t+=['','可预测性与欺诈相关性不同：容易重构的字段也可能与欺诈无关，难重构字段也可能含关键异常。字段宽度先验及0.25下限都有独立消融；本报告保留退步结果，不按测试分数翻转异常方向、筛种子或改变权重。完整逐字段skill、MSE基线和权重见[weights.json](weights.json)。','','## 逐行基线参考（同数据／划分，不是本仓库新训练）','','BankSim／S-FFSD参考继承已核验的同协议历史结果；CardSim的9个HBOS／LOF／VAE历史运行重新核验保存分数、阈值和指标。HIMVH使用额外原生关联信息，信息预算不完全等同于逐行方法。','','| 数据集 | 方法 | AP | POT F1 | q99 F1 |','|---|---|---:|---:|---:|']
    for d in TASKS:
        for m in ['hbos','lof','vae']:
            r=next(r for r in baselines if (r['dataset'],r['method'],r['operating_point'])==(d,m,'pot'));q=next(r for r in baselines if (r['dataset'],r['method'],r['operating_point'])==(d,m,'quantile'));t.append('| '+d+' | '+m+' | '+ms(r,'average_precision')+' | '+ms(r,'f1')+' | '+ms(q,'f1')+' |')
    t+=['','## 数据与评价边界','','训练／校准不读取真实标签；训练目标最多8192行，校准最多100000行，测试全集，种子42/43/44。未知标签-1只在最终评价排除，不按正常标签清洗源数据。m重建抽样训练视图，cm用完整原生训练上下文，但损失目标相同。每阶段独立构图，没有跨分区边；同阶段未来交易仍能作为上下文，所以是离线检测。构图键可能携带被遮蔽字段的结构信息，不能声称所有类别线索都被移除。','','POT q=.85／risk=.01／min20，失败N/A，独立q99；不做point-adjust。本文F1为异常类F1，论文报告macro-F1，数值不直接比较；AP与梯形PR-AUC分开记录。校准CDF、权重和POT共用校准样本，没有有限样本误报保证；高异常占比下只看F1容易误判。','','| 数据集 | 已知测试异常占比＝常数AP | 旧／独立范围 |','|---|---:|---|']
    for d in TASKS:t.append('| '+d+' | '+fmt(sx[d,'cm','field','pot']['prevalence'])+' | '+('一个独立模拟来源' if d=='cardsim' else '旧测试回顾性')+' |')
    t+=['','## 核验和复现','','18个模型均验证checkpoint张量加载、原采样索引、训练预处理、语义字段划分及固定CPU稀疏邻域探针；误差容差rtol5e-4/atol2e-5以覆盖原CUDA散射和CPU计算差异。全部评分使用归档的逐字段误差重新计算，独立按unique/count重建经验CDF，重算常量基线与权重，再独立重拟合POT/q99及全部指标。BankSim／S-FFSD的48组既有field/flat/tail/tail_flat结果与上一轮指标完全一致。跨设备端到端重新推理可能有微小数值差异，不能保证经验CDF分数及阈值边界报警逐位相同。','','[核验记录](verification.json) · [逐种子144配置](per_seed.json) · [288条操作点](metrics.csv) · [96条汇总](summary.csv) · [配对消融](paired_effects.csv) · [历史指标一致性](historical_metric_identity.json) · [父模型哈希](parent_models.json)。原检查点来自历史CQ归档，未将训练时间说成本次耗时。','','快速命令及重新训练入口见[项目README](../../README.md)。三个预训练示例统一选择cm/seed42，未按最佳种子筛选。其他种子可从原CQ归档恢复或用公开数据及本包重新训练；没有把私人归档路径写成公开可下载地址。','','## Plain English','','We re-evaluated144 scoring configurations on18 frozen HIMVH zero-label models and audited288 operating points, with zero new financial neural-network fits. Existing training artifacts and this new scoring run are distinguished. All prescribed ablations and negative effects are retained;the fixed skill-weight primary is not swapped by dataset. CardSim is one independently generated synthetic world,not external real-bank validation or zero-shot transfer. Paper supervised results and macro-F1 are not reproduced by this reconstruction adaptation.','']
    (a.output/'README_CN.md').write_text('\n'.join(t));print('Published144 score configs,288 operating rows,96 summaries and3 fixed pretrained examples')
if __name__=='__main__':main()
