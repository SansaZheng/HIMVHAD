# Evidence map

| Claim | Evidence | Limit |
|---|---|---|
| Standalone zero-label package,not supervised original | model.py/network.py;docs/reproduction_CN.md | Supervised label pathways/classifier removed;not paper score replication |
| No fitting/calibration labels parsed | 5-variant absent-label tests;prediction-label guard;frozen evaluation guard | Data preparation can write labels;test labels used only in evaluation |
| Parent math preserved | results/parent_equivalence.json;dense/ego forward+gradient tests | 4 synthetic control fits;not fresh financial training |
| 18 frozen models/144 score settings/288 operating points verified | results/frozen/verification.json;per_seed.json | Saved components re-scored;CPU probes with stated tolerance |
| 48 prior configurations exact | results/frozen/historical_metric_identity.json | Historical old-data reuse is retrospective |
| One independent synthetic source evaluated | fixed source protocol,parent training manifest,CardSim results | One world,not real-bank external validation or zero-shot |
| Skill weights not stable improvement | paired_effects.csv;summary.csv;full report | No per-dataset primary substitution;3seeds not significance |
| 3 pretrained examples and wheel work | release_verification.json;pretrained provenance;CLI tests | Fixedcm/42,not best seeds;wheel excludes Git-hosted weight examples |


## 2026-11-23 隐藏边界融合扩展

已实现冻结HIMVH隐藏状态、CNN代理边界、候选维度、层注意力KAN、解析近似与探针方向诊断。零真实标签；训练误差代理不等于欺诈标签。3数据集×3骨干种子，9组、81个正式配置，另保留81配置初跑。原版数据划分、测试行、预处理和骨干检查点不变。

BankSim原AP16.53%→6.37%（-10.16pp），S-FFSD35.19%→38.28%（+3.09pp），CardSim2.20%→1.66%（-0.54pp）。主方法POT-F1三组均下降；9/9组未达到k≤10、1pp代理BA容差判据。解析R²0.9960–0.9985，当前损失不能主要归因于解析失真。主方法未因测试最优消融而重命名；默认原版保留。

证据：results/knob_20261123/README_CN.md、metrics.csv、summary.csv、paired_differences.csv、verification.json、cq_receipts.json。原始逐点分数及全种子模型保存在CQj-1123-c26354；初跑j-1123-21f2ed独立归档。9个唯一冻结骨干，新骨干拟合0；正式新增神经头拟合126（含k搜索），不能说新NN训练0次。results.csv的new_financial_fit只表示该评分涉及新增头，不代表独立拟合计数，不得直接求和。

35项本地及CQ测试、轮安装重载、225评分前文件哈希和源码SHA、81配置/162工作点独立回放均通过。历史9组基线指标完全一致；独立跨平台重评分末位差最大8.882e-16，按1e-12容差核验；同轮融合恒等式与两次运行54条不变路径仍按位一致。CardSim16021行9设置本机预测不读取真实标签，两种阈值报警变化均0；主方法最大分数差0.001546，跨版本差异保留。

论文原任务需要正确性标签；此处是代理误差目标的新适配，不是原文监督效果复刻。内部代理验证在原训练分区且骨干已见这些特征；无新独立真实银行留出集。方向干预仅在归一化探针状态空间，未验证完整图网络的金融因果作用。


## Elliptic，2026-11-24

通过预先固定的 Elliptic 验证，三种子平均 AP 从 3.14% 降至 3.02%，配对变化 −0.11 个百分点。POT-F1 和 q99-F1 从 0.17% 降至 0%。三组压缩代理判据都通过，未转化为检测收益。8,841 笔有已知标签，其中 408 笔为非法交易，另有 27,983 笔测试交易参与上下文与评分。真实标签在全体分数和阈值冻结后打开。

[完整结果与限制](../../results/elliptic_20261124/README_CN.md) · [评分与数据核验](../../results/elliptic_20261124/verification.json)。正式 CQ 作业 j-1124-5e4973。三个种子沿用同一套训练预算和评分配置。基线与融合共用各自种子的新骨干。

English: On Elliptic, the fixed fusion lowered mean AP and produced zero true positives at both fixed operating points across all seeds. All three searches met the proxy compactness criterion. This supports separating proxy fidelity from detection utility.
