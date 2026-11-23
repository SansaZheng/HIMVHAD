# HIMVHAD

**HIMVH 的独立零标签金融交易异常检测适配，可安装、训练、重评分和预测。** 使用遮蔽字段重构学习，不把真实训练／校准标签送入模型。它与[原论文](https://arxiv.org/html/2601.11073v3)的监督分类器有明确区别；本仓库不宣称复刻论文监督成绩。

[中文方法说明](docs/reproduction_CN.md) · [完整结果与消融](results/frozen/README_CN.md) · [数据来源](docs/data_sources_CN.md) · [English](README_EN.md)

## 隐藏边界融合扩展（2026-11-23）

已融入用户提供论文的CNN探针、候选维度筛选、层注意力KAN、解析近似和方向诊断，采用**零真实标签适配**：代理目标来自训练重构误差。新增头与原冻结HIMVH组合，保留原始数据划分、测试行、POT/q99评价标准。

[融合机制与复现](docs/mechanistic_boundary_CN.md) · [完整前后对比与消融](results/knob_20261123/README_CN.md) · [逐种子指标](results/knob_20261123/metrics.csv) · [核验记录](results/knob_20261123/verification.json)

主方法预先固定为 `kan_fused`，以下为3个骨干种子的AP均值（%）；原版指已有 `cm_tail_flat` 配置：

| 数据集 | 融合前AP | 融合后AP | 差值/百分点 |
|---|---:|---:|---:|
| BankSim | 16.53 | 6.37 | -10.16 |
| S-FFSD | 35.19 | 38.28 | +3.09 |
| CardSim | 2.20 | 1.66 | -0.54 |

**本轮未显示稳定改进。** S-FFSD的AP提高，但三个数据集的POT-F1均下降；9/9组未在k≤10内达到近最优压缩判据。默认原版流程保留，扩展通过独立命令调用。全部9项消融、负结果、初跑及随机流修正都保留，未按测试成绩替换主方法。

35项本地及CQ测试通过，81组最终设置／162个工作点独立核验通过；初跑81组另存。三组固定seed42预训练适配器随结果提供，不按最佳种子筛选。完整金融数据为同一批回顾性测试，未新增真实银行外部验证。

完成下方安装后，可直接运行预训练扩展：

```bash
python -m himvhad fetch-benchmark --root data/benchmark --datasets cardsim
mkdir -p runs/knob-parent-cardsim42
tar -xzf experiments/knob_inputs/cardsim-42.tar.gz -C runs/knob-parent-cardsim42
python -m himvhad knob-predict \
  --parent runs/knob-parent-cardsim42 \
  --adapter results/knob_20261123/runs/cardsim/42 \
  --data data/benchmark/cardsim --output runs/knob-cardsim-prediction
```

该命令只使用冻结模型／校准器，不读取真实标签。重新训练新增头及全部消融的命令见方法说明。

Elliptic 外部数据验证已完成。三种子平均 AP 从 3.14% 降到 3.02%，POT-F1 从 0.17% 降到 0.00%。四个数据集的变化、27 组评分及复现命令见 [Elliptic 中文报告](results/elliptic_20261124/README_CN.md)。

## 安装与CPU演示

```bash
git clone https://github.com/SansaZheng/HIMVHAD.git
cd HIMVHAD
python -m pip install -e '.[test]'
python -m himvhad demo --output runs/demo --steps 2
python -m pytest -q
```

支持Python3.10及以上，CPU演示不需要GPU。`demo`是合成流程检查，不是金融成绩；输出目录必须是新目录。完整金融训练是后续显式命令，安装和演示不会自动启动它。

## 直接使用随仓库提供的模型

提供BankSim、S-FFSD、CardSim三个**固定cm/seed42**历史检查点及其训练缩放器、无标签校准器和阈值，未按最佳种子筛选。原始数据另行下载、SHA256核验；无需登录数据平台。

```bash
python -m himvhad fetch-benchmark --root data/benchmark --datasets cardsim
python -m himvhad predict \
  --run examples/pretrained/cardsim \
  --data data/benchmark/cardsim \
  --score tail \
  --output runs/cardsim-prediction
```

输出`predictions.npz`包括分数和可用阈值的布尔报警，`prediction_manifest.json`记录来源；预测不读取真实标签。POT拟合失败时不产生`pot`报警数组，失败原因保留在模型目录对应评分的`thresholds.json`，不会用q99冒充POT。

如需对已知测试标签评价，使用冻结阈值：

```bash
python -m himvhad evaluate \
  --scores runs/cardsim-prediction/predictions.npz \
  --labels data/benchmark/cardsim/test.y.i8 \
  --thresholds examples/pretrained/cardsim/tail/thresholds.json \
  --output runs/cardsim-prediction/metrics.json
```

## 重新训练相同设置

```bash
python -m himvhad fetch-benchmark \
  --root data/benchmark --datasets banksim s_ffsd cardsim

python -m himvhad run \
  --data data/benchmark/banksim \
  --output runs/banksim \
  --variants m cm --seeds 42 43 44 \
  --steps 800 --device cuda \
  --scores field flat tail tail_flat skill_tail skill_raw skill_equal_tail skill_hard_tail
```

没有CUDA时使用`--device cpu`。换数据集只改`--data`和`--output`。默认hidden256、3层、窗口4、每步128目标、8192训练目标、最多100000校准样本；median/IQR缩放只拟合训练特征。固定POT q0.85/risk0.01/min20及独立q99；未知标签只在最终评价排除，不以正常标签筛训练数据。

`m`重建抽样训练视图；`cm`保留完整原生训练上下文，承担损失的交易目标相同。评分和阈值持久化后才解析测试标签。每阶段独立构图，使用同阶段完整上下文，属于离线检测，不能当作严格逐笔因果报警。

`legacy/f00/c`也可运行以审计历史行为，边界见方法说明；历史f00/c原评分对应`flat`。`mean_scale/median_scale/tail_max/tail_corr`是其他可选评分。`skill_*`是本项目扩展，不属于论文原方法，现有结果未显示稳定收益。

若本机已有父项目文件，可离线复用：

```bash
python -m himvhad fetch-benchmark --root data/benchmark \
  --datasets banksim s_ffsd cardsim \
  --source-local /path/to/Financial_unsuperviesdAD
```

## 冻结模型只重评分

```bash
python -m himvhad rescore \
  --parent runs/banksim/cm/42 \
  --data data/benchmark/banksim \
  --output runs/banksim-rescored \
  --scores field flat tail tail_flat skill_tail
```

该命令复用检查点及`components.npz`，不训练网络；skill评分还需要原训练／校准特征。随仓库的轻量预训练示例供`predict`使用，未捆绑大批逐点误差；要使用`rescore`，请使用完整训练输出或原CQ归档。

## 自己的交易CSV

准备[配置JSON](docs/example_csv_config.json)，明确时间、原生视图键、数值字段、类别哈希维数和可选测试标签映射：

```bash
python -m himvhad prepare-csv --csv transactions.csv \
  --config docs/example_csv_config.json --output data/custom
python -m himvhad run --data data/custom --output runs/custom --steps 800
```

按完整时间戳60/20/20切分；自定义数据只写测试标签，训练／校准标签文件不是必要输入。数值字段须有限。新数据自己的训练／校准分区用于建模，评价测试分区；不声称旧模型零样本跨来源迁移。

## Python接口

```python
from himvhad import HIMVHAnomalyDetector
from himvhad.preprocessing import Preprocessor

# x_train/x_test为数值特征；views_*为原生键形成的超边列表。
pre = Preprocessor().fit(x_train[train_indices])
model = HIMVHAnomalyDetector(width=61, task="banksim", steps=800)
model.fit(pre.transform(x_train), views_train, target_rows=train_indices)
field_errors = model.components(pre.transform(x_test), views_test)
model.save("model.pt")
restored = HIMVHAnomalyDetector.load("model.pt")
```

自定义字段使用`field_groups`明确划分；支持多于一种视图数，这是接口扩展。校准评分见`FieldScorer/SkillScorer`，完整数据流程以CLI为例。

## 基础版本核验与结果（2026-11-21）

基础提取版本当时复用**18个冻结金融模型，重新计算144组评分／288条操作点**，没有新金融神经网络训练。旧数据48组既有评分指标完全一致；新源码训练与原端口有数值等价控制测试。三个数据源及所有消融、负结果、原始模型SHA和核验记录见[中文结果报告](results/frozen/README_CN.md)。

BankSim/S-FFSD为旧测试回顾性评估；CardSim为一个预先固定参数后生成的独立模拟世界，不能替代真实银行外部验证。重构可预测性不是欺诈相关性，新增权重没有稳定超过已有评分对照。本文F1是异常类F1，论文报告macro-F1，两者不直接比较。跨设备推理可能使阈值边缘样本有微小差异；重新评分与端到端推理分别记录。

基础版本当时的27项本地测试、wheel及实际演示已通过；新增扩展后已通过35项本地和CQ测试。GitHub CI已配置；发布核对时API尚无远端运行记录，未标记为CI通过。

完整文件索引、来源、命令和环境见[发布核验](results/release_verification.json)与[工作流状态](paper/workflow/current.md)。

## 来源与许可

代码MIT；独立实现源自[SansaZheng/Financial_unsuperviesdAD](https://github.com/SansaZheng/Financial_unsuperviesdAD)，文件SHA见[迁移清单](PORTING_MANIFEST.json)。数据许可证单列，未重新许可为MIT。引用原论文及本仓库可参考[CITATION.cff](CITATION.cff)。
