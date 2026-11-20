# HIMVHAD

HIMVH 的独立零标签金融异常检测适配。训练目标为遮蔽字段重构，真实训练／校准标签不参与学习；该实现不是原论文的监督分类器，也不宣称复刻原论文分数。

```bash
python -m pip install -e '.[test]'
python -m himvhad demo --output runs/demo --steps 2
python -m pytest -q
```

完整中文方法说明、实验入口和来源核验报告正在本次整理中。上游实现与已核验结果来自 [Financial_unsuperviesdAD](https://github.com/SansaZheng/Financial_unsuperviesdAD)。原论文：[arXiv:2601.11073v3](https://arxiv.org/html/2601.11073v3)。

Independent zero-label adaptation: masked-field reconstruction instead of the paper's supervised classification. The synthetic demo is a pipeline check, not financial benchmark evidence.
