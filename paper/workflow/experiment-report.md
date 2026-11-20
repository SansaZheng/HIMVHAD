# 中文／English

中文：HIMVHAD将现有零标签HIMVH适配整理成独立包，提供原生超图、一致遮蔽重构、精确邻域训练及字段评分消融。27测试通过，18个旧模型重新评分144配置／288操作点，未新增金融NN训练。CardSim原训练归档已完成并被复用；本次评价保持其固定方案。新权重无稳定提升、独立模拟数据上存在退步，结果完整保留。旧BankSim/SFFSD分析为回顾性，CardSim不是外部真实银行验证。

English:The standalone package retains the zero-label masked-reconstruction adaptation and native hypergraph modules.27tests and18 frozen-model audits pass;144 scoring settings/288 operating points were evaluated without new financial NN fitting. Historical imports,frozen re-scoring and synthetic demo are distinct. Skill weighting is not a consistent improvement. The single CardSim synthetic world is not real-bank external validation.
