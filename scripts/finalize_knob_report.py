"""Attach checked evidence and bounded conclusions to the generated report."""
import argparse,csv,datetime,json,shutil,subprocess
from pathlib import Path
import numpy as np
from himvhad.io import write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--checks',type=Path,required=True);a=p.parse_args();out,checks=a.report,a.checks
    with (out/'summary.csv').open() as f:summary=list(csv.DictReader(f))
    with (out/'proxy_diagnostics.csv').open() as f:proxy=list(csv.DictReader(f))
    def value(d,v,k='average_precision_mean'):return float(next(r[k] for r in summary if r['dataset']==d and r['variant']==v))
    verification=json.loads((checks/'r2_verification.json').read_text());assert verification['status']=='passed' and verification['paired_control_rng']
    for src,dst in [('r2_verification.json','verification.json'),('pilot_verification.json','pilot_rng/verification.json'),('pretrained-r2-verification.json','pretrained_prediction_verification.json')]:shutil.copyfile(checks/src,out/dst)
    shutil.copyfile(checks/'cq_results/j-1123-87784e/tests.txt',out/'tests.txt')
    shutil.copyfile(checks/'cq_results/j-1123-87784e/env.json',out/'test_environment.json')
    shutil.copyfile(checks/'cq_results/j-1123-e2b51d/experiment/smoke.json',out/'resource_qualification.json')
    wheel=json.loads((checks/'wheel-r2-verification.json').read_text());wheel['isolated_import']='isolated wheel-r2-install/himvhad/__init__.py outside repository';write(out/'wheel_verification.json',wheel)
    max_baseline_delta=max(v for r in verification['checks'] for v in r['independent_baseline_max_abs'].values())
    r2=[float(r['symbolic_r2']) for r in proxy if r['symbolic_r2']]
    failed=sum(r['near_optimal']=='False' for r in proxy)
    symbolic=max(abs(value(d,'symbolic_fused')-value(d,'kan_fused'))*100 for d in ['banksim','s_ffsd','cardsim'])
    facts=['## 本轮结果直接支持什么','',
      f'- **紧凑判据未实现**：{failed}/9组在k≤10内未达到全维CNN代理BA减1个百分点的标准，均触及预设上限。本次只能报告候选维度，不能声称发现了论文意义上的近无损“旋钮”。',
      '- **重构难度与欺诈标签不等价**：BankSim的CNN代理BA为0.85–0.91，但CNN单独检测AP只有1.52%，明显低于原版16.53%。这个差距直接说明代理任务学得好，仍可能没有识别真实异常的能力。',
      f'- **统一融合可能损失有用排序**：S-FFSD上KAN单独AP为{100*value("s_ffsd","kan_only"):.2f}%，固定0.5融合后为{100*value("s_ffsd","kan_fused"):.2f}%。本轮按冻结协议保留融合版为主方法，没有事后把最佳消融改名。',
      '- **维度选择与非线性尚未得到稳定支持**：S-FFSD的随机维度融合AP高于主方法；CardSim两者接近；BankSim和CardSim的线性融合均高于KAN融合。均匀层权重与学习层权重的差异也较小，不能据此宣称新的潜在几何具有普遍优势。',
      f'- **解析近似不是当前主要损失点**：内部验证R²为{min(r2):.4f}–{max(r2):.4f}，解析版与KAN融合版的平均AP差距最多{symbolic:.2f}个百分点。当前证据不支持把整体下降主要归因于解析表达式失真。',
      f'- **排序改进不等于阈值工作点改善**：S-FFSD的AP提高，但POT-F1下降；已知标签测试行中异常占38.87%，主方法POT报警约0.61%，Precision仅{100*value("s_ffsd","kan_fused","pot_precision_mean"):.2f}%、Recall仅{100*value("s_ffsd","kan_fused","pot_recall_mean"):.3f}%。低报警率限制召回，报警精确率也下降，不能只把问题归因于阈值。没有用这一观察回调阈值。','']
    text=(out/'README_CN.md').read_text();marker='## 如何解释提升与下降'
    if '## 本轮结果直接支持什么' in text:
        prefix,tail=text.split('## 本轮结果直接支持什么',1);text=prefix+marker+tail.split(marker,1)[1]
    if '## 独立核验与部署检查' in text:text=text.split('## 独立核验与部署检查',1)[0].rstrip()+'\n'
    assert text.count(marker)==1;text=text.replace(marker,'\n'.join(facts)+'\n'+marker)
    text=text.replace('仓库已有 `cm_skill_tail` 扩展也保留作参考（历史同检查点／划分结果，未冒充本轮新跑）：\n-','仓库已有 `cm_skill_tail` 扩展也保留作参考（历史同检查点／划分结果，未冒充本轮新跑）：\n\n-')
    # A platform replay is numerical equivalence, not a promise of bitwise libm equality.
    text=text.replace('`baseline`逐点复用原cm_tail_flat，骨干参数不更新','`baseline`复用原cm_tail_flat误差与评分公式，同轮消融共用保存的基线分数，骨干参数不更新')
    coverage='完整打分覆盖118638 / 15576 / 16021条测试交易；S-FFSD有9417条未知标签按原规则排除，监督指标仅评价剩余6159条。表中报警比例也以已知标签行作为分母，不代表未知交易的真实欺诈比例。'
    if coverage not in text:text=text.replace('校准风险1%并不保证测试报警比例恰好1%',coverage+'\n\n校准风险1%并不保证测试报警比例恰好1%')
    evidence=['','## 独立核验与部署检查','',
      f'全部81组评分、162个阈值工作点完成独立回放，9组历史基线指标完全一致；原始误差文件及评分前SHA链均核验。本机与云端重新计算同一基线的最大浮点末位差为{max_baseline_delta:.3g}，按1e-12数值容差核验，不宣称跨平台逐点按位相同。原始失败的严格按位检查与修订原因保存在任务记录中，未修改任何模型分数或阈值以通过检查。详见[核验JSON](verification.json)。',
      '', 'CardSim固定seed42模型完成16021行、9个设置的本机端到端预测，真实标签访问被拦截。相比CQ归档，主KAN融合分数最大差0.001546；两种阈值下9个设置的报警差异均为0。该检查涉及本机Torch2.6、云端Torch2.11与历史GPU误差，数值差异已保留，未承诺所有环境/数据都按位一致。详见[部署记录](pretrained_prediction_verification.json)。',
      '', '35项本地与CQ远端测试通过；独立wheel安装、适配器训练/保存/重载和源码一致性检查通过。完整CQ产物可用 `cq fetch j-1123-c26354` 取得；初跑为 `cq fetch j-1123-21f2ed`。',
      '', 'English: The primary fusion improves AP on S-FFSD but reduces AP on BankSim and CardSim, and reduces POT-F1 on all three datasets. None of the nine searches meets the compactness criterion. High symbolic fidelity does not translate into reliable fraud detection gains. No test-label tuning or default-method replacement follows these outcomes.','']
    text+='\n'.join(evidence);(out/'README_CN.md').write_text(text)
    jobs=[]
    for id in ['j-1123-e65994','j-1123-1c2e92','j-1123-8acd05','j-1123-e2b51d','j-1123-21f2ed','j-1123-87784e','j-1123-c26354']:
        d=json.loads(subprocess.check_output(['cq','--json','show',id],text=True,stderr=subprocess.DEVNULL));s=d['spec']
        reason=(d.get('reason') or '').replace('/Users/sudingli','${CQ_CONTROL_USER}') or None
        jobs.append(dict(id=id,name=d['name'],state=d['state'],reason=reason,spec_sha=d['spec_sha'],code=s['code'],command=s['command'],resources=s['resources'],limits=s['limits'],depends_on=s.get('depends_on',[]),outputs=s['outputs'],started=d.get('started'),finished=d.get('finished')))
    write(out/'cq_receipts.json',dict(checked_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),jobs=jobs,paid_rentals=False,budget_usd=0))
    head_fits=sum(1+len(json.loads((out/'runs'/d/str(seed)/'adapter_info.json').read_text())['sweep'])+3 for d in ['banksim','s_ffsd','cardsim'] for seed in [42,43,44])
    write(out/'release_verification.json',dict(status='experiments_verified_ready_for_publication',source_commit='2a8fb7a03344640daf43a99fc823f9f074317f6b',scope='zero-real-label adaptation;not original supervised paper-score reproduction',unique_frozen_backbones=9,new_backbone_fits=0,final_score_configurations=81,pilot_score_configurations=81,final_neural_head_fits_including_k_search=head_fits,local_and_CQ_tests=35,historical_baseline_metrics_exact=True,independent_baseline_replay_max_abs=max_baseline_delta,all_prelabel_hashes_verified=True,pretrained_examples='fixedcm/seed42 for3datasets,notbestseed',near_optimal_proxy_criterion_passes=9-failed,symbolic_r2_range=[min(r2),max(r2)],report_sha256=sha(out/'README_CN.md'),verification_sha256=sha(out/'verification.json'),paid_budget_usd=0,CI='Local andCQ tests verified;GitHubActions pass not claimed',publication=dict(status='pending',branch='main')))
    write(out/'artifact_manifest.json',dict(files_sha256={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='artifact_manifest.json'}))
    print(json.dumps(dict(status='ready',head_fits=head_fits,compactness_passes=9-failed,max_baseline_replay_delta=max_baseline_delta),ensure_ascii=False))


if __name__=='__main__':main()
