# 中文／English

中文：HIMVHAD将现有零标签HIMVH适配整理成独立包，提供原生超图、一致遮蔽重构、精确邻域训练及字段评分消融。27测试通过，18个旧模型重新评分144配置／288操作点，未新增金融NN训练。CardSim原训练归档已完成并被复用；本次评价保持其固定方案。新权重无稳定提升、独立模拟数据上存在退步，结果完整保留。旧BankSim/SFFSD分析为回顾性，CardSim不是外部真实银行验证。

English:The standalone package retains the zero-label masked-reconstruction adaptation and native hypergraph modules.27tests and18 frozen-model audits pass;144 scoring settings/288 operating points were evaluated without new financial NN fitting. Historical imports,frozen re-scoring and synthetic demo are distinct. Skill weighting is not a consistent improvement. The single CardSim synthetic world is not real-bank external validation.


## 2026-11-23 隐藏边界融合扩展

已实现冻结HIMVH隐藏状态、CNN代理边界、候选维度、层注意力KAN、解析近似与探针方向诊断。零真实标签；训练误差代理不等于欺诈标签。3数据集×3骨干种子，9组、81个正式配置，另保留81配置初跑。原版数据划分、测试行、预处理和骨干检查点不变。

BankSim原AP16.53%→6.37%（-10.16pp），S-FFSD35.19%→38.28%（+3.09pp），CardSim2.20%→1.66%（-0.54pp）。主方法POT-F1三组均下降；9/9组未达到k≤10、1pp代理BA容差判据。解析R²0.9960–0.9985，当前损失不能主要归因于解析失真。主方法未因测试最优消融而重命名；默认原版保留。

证据：results/knob_20261123/README_CN.md、metrics.csv、summary.csv、paired_differences.csv、verification.json、cq_receipts.json。原始逐点分数及全种子模型保存在CQj-1123-c26354；初跑j-1123-21f2ed独立归档。9个唯一冻结骨干，新骨干拟合0；正式新增神经头拟合126（含k搜索），不能说新NN训练0次。results.csv的new_financial_fit只表示该评分涉及新增头，不代表独立拟合计数，不得直接求和。

35项本地及CQ测试、轮安装重载、225评分前文件哈希和源码SHA、81配置/162工作点独立回放均通过。历史9组基线指标完全一致；独立跨平台重评分末位差最大8.882e-16，按1e-12容差核验；同轮融合恒等式与两次运行54条不变路径仍按位一致。CardSim16021行9设置本机预测不读取真实标签，两种阈值报警变化均0；主方法最大分数差0.001546，跨版本差异保留。

论文原任务需要正确性标签；此处是代理误差目标的新适配，不是原文监督效果复刻。内部代理验证在原训练分区且骨干已见这些特征；无新独立真实银行留出集。方向干预仅在归一化探针状态空间，未验证完整图网络的金融因果作用。

English: A frozen-backbone, zero-label CNN/KAN boundary extension was implemented and evaluated onnine paired runs. The prespecified fusion improves AP only onS-FFSD and reduces POT-F1 onallthree datasets. None ofthe nine searches meets the compactness criterion. High symbolic fidelity does not establish fraud correctness. Allnegative results,the RNG-confounded pilot andthe corrected ablations are preserved.
