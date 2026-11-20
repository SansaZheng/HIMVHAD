# 数据来源与许可证

本仓库不捆绑原始金融数据。`fetch-benchmark`从父项目固定Git提交下载公开封存件，逐文件核验SHA256，再准备原生视图键；源记录在[benchmark_sources.json](../himvhad/benchmark_sources.json)。可用`--source-local /path/to/Financial_unsuperviesdAD`复用本机已有文件，不访问网络。

- BankSim：原提供方https://www.kaggle.com/datasets/ealaxi/banksim1；父项目保存的来源声明为CC-BY-NC-SA-4.0。[原声明](https://github.com/SansaZheng/Financial_unsuperviesdAD/blob/f0a6c7fa00eb54fba0f76b421cc11f66b9b536fe/data/raw/banksim/CC-BY-NC-SA-4.0.txt)。
- S-FFSD：[原数据仓库](https://github.com/AI4Risk/antifraud)。源仓库为GPL-3.0；此前没有确认独立数据许可证。此处记录来源，不将其重新许可为MIT。
- CardSim：[美联储官方模拟器](https://www.federalreserve.gov/econres/feds/cardsim-a-bayesian-simulator-for-payment-card-fraud-detection-research.htm)。使用已封存官方代码和公开调查输入，seed2026111707、1024付款者、90天、payer_payee_factor1，其余默认。80107交易、26封存文件。生成前方案已固定；标签由模拟器规则产生。

BankSim/S-FFSD下载缓存合计约65MiB，解压及视图键需要额外空间；CardSim输入归档约3MiB。下载原始键列只是构造视图；训练与校准不读取目标标签。代码MIT许可与数据来源声明分别保留。
