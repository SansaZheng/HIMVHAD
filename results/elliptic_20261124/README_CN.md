# Elliptic 独立数据验证

使用 Elliptic v1 的真实比特币交易记录。训练、校准和测试分别包含时间步 1–30、31–41、42–49。
模型使用 93 个交易本地特征，时间用于切分。匿名特征按原列顺序分为六组。三个图视图来自直接资金流、共同上游和共同下游。后两个视图按交易 ID 排序连接相邻项，边仅来自同一阶段内的交易快照。模型基于完整快照进行离线检测。

训练 123,287 笔，校准 43,658 笔，测试 36,824 笔。已知测试标签 8,841 笔，其中非法交易 408 笔。其余 27,983 笔参与图上下文与打分，指标只统计已知标签。

[官方数据](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) · [数据论文](https://arxiv.org/abs/1908.02591) · [冻结协议](../../paper/workflow/prompts/005-elliptic-holdout.md)

首个 CQ 作业访问官方链接时收到 404，未开始训练。完成实验的作业使用 PyTorch Geometric 镜像，三个 CSV 的解压后 SHA256 与官方 v1 归档逐一相等。复现可使用官方归档或经过校验的镜像。
[PyG 下载代码](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/elliptic.html)

三个新骨干种子各训练 800 步，每个种子的融合与消融共用该种子的骨干。CNN/KAN 只学习训练数据的重构误差代理目标，主融合权重固定为 0.5。三个种子的所有分数和阈值保存后才打开标签文件。下表为种子均值 ± 样本标准差，单位为百分比。配对差值使用未舍入的分数计算。

| 评分 | AP | 梯形 PR-AUC | ROC-AUC | POT-F1 | q99-F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 3.14 ± 0.15 | 3.13 ± 0.15 | 33.04 ± 3.62 | 0.17 ± 0.30 | 0.17 ± 0.29 |
| cnn_only | 3.15 ± 0.20 | 3.14 ± 0.19 | 32.96 ± 4.15 | 0.10 ± 0.17 | 0.10 ± 0.17 |
| cnn_fused | 3.06 ± 0.13 | 3.05 ± 0.13 | 30.94 ± 3.80 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| kan_only | 3.04 ± 0.07 | 3.03 ± 0.07 | 30.10 ± 1.87 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| kan_fused | 3.02 ± 0.06 | 3.02 ± 0.06 | 29.62 ± 1.79 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| random_fused | 3.22 ± 0.44 | 3.20 ± 0.44 | 33.25 ± 10.04 | 0.21 ± 0.37 | 0.21 ± 0.37 |
| uniform_fused | 3.02 ± 0.07 | 3.01 ± 0.07 | 29.56 ± 1.87 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| linear_fused | 3.11 ± 0.25 | 3.10 ± 0.24 | 31.74 ± 5.82 | 0.10 ± 0.18 | 0.00 ± 0.00 |
| symbolic_fused | 3.03 ± 0.07 | 3.02 ± 0.07 | 29.77 ± 1.92 | 0.08 ± 0.14 | 0.08 ± 0.15 |

主融合的平均 AP 变化为 -0.11 个百分点，1/3 个种子的 AP 高于原版。全部设置中平均 AP 最高的是 `random_fused`。已知测试标签中非法交易占 4.61%，对应随机排序的 AP 参考水平。

| 评分 | POT 精确率 | POT 召回率 | q99 精确率 | q99 召回率 |
|---|---:|---:|---:|---:|
| baseline | 0.18 ± 0.31 | 0.16 ± 0.28 | 0.17 ± 0.30 | 0.16 ± 0.28 |
| kan_fused | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

融合减原版的配对变化（百分点）

| 种子 | ΔAP | ΔPOT-F1 | Δq99-F1 |
|---|---:|---:|---:|
| 42 | -0.23 | +0.00 | +0.00 |
| 43 | -0.11 | -0.51 | -0.51 |
| 44 | +0.00 | +0.00 | +0.00 |

与前三个数据集一起看融合方向，差值都以各自的原版为基准（百分点）。

| 数据集 | ΔAP | ΔPOT-F1 | Δq99-F1 |
|---|---:|---:|---:|
| BankSim | -10.16 | -15.00 | -14.81 |
| S-FFSD | +3.09 | -0.46 | -0.11 |
| CardSim | -0.54 | -0.55 | -0.49 |
| Elliptic | -0.11 | -0.17 | -0.17 |

内部代理验证有 3/3 组达到 10 维以内、距全维 CNN 平衡准确率至多 1 个百分点的标准。
正式运行使用 cpu，三个种子的拟合和评分共用时 31.4 分钟。原基线逐点一致，27 组分数和阈值通过独立回放。

[逐种子指标](metrics.csv) · [配对差值](paired_differences.csv) · [压缩诊断](adapter_diagnostics.json) · [核验记录](verification.json)

各数据集的输入特征、图结构和标签覆盖不同。Elliptic 的结果适用于这组公开比特币快照的已知标签样本。三组配对结果反映初始化差异，跨数据来源仍需结合此前三个数据集的结果判断。

主融合的两个阈值共覆盖三个种子和六个工作点。每个工作点产生了 212–379 条已知标签报警，非法交易命中均为 0。原版在种子 43 的两个工作点各命中 2 笔，其余种子为 0。主融合的 AP 和 ROC-AUC 同时下降，两版 AP 都低于非法交易占比 4.61%，排序表现也较弱。

三个种子分别选中 8、2、10 个维度，代理验证的平衡准确率达到 87.86%、88.26%、89.42%。这与真实非法交易检测的低 AP 并存，说明重构误差代理边界的压缩保真度与检测效用应分开评价。随机维度融合在本数据中取得最高均值 AP 3.22%，亦低于 4.61% 的参考水平。

正式 CQ 作业为 `j-1124-5e4973`。通过 `cq fetch j-1124-5e4973` 可取回归档中的逐笔分数、阈值和模型。公开报告保留指标、配对差值及文件哈希。 [作业记录](cq_receipts.json)

复现实验

```bash
python -m pip install -e '.[test]'
curl -fL 'https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic-data-set?datasetVersionNumber=1' -o elliptic.zip
PYTHONPATH=. python scripts/run_elliptic_holdout.py --archive elliptic.zip --output runs/elliptic --device cpu
```

English: This experiment evaluates the prespecified zero-label fusion on temporally held-out Elliptic Bitcoin transactions using three matched backbones and the same nine scoring settings.
