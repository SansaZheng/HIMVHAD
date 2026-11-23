# 005 Elliptic held-out dataset validation

Local task specification. Codex and CQ. Configuration revision 2, same zero-label method, no venue. Started 2026-11-24 Asia/Singapore.

Validate the original and CNN/KAN fusion on Elliptic, a new dataset for this project. Pin the official public release. Retain the 93 transaction-local features excluding time, use time only for whole-snapshot chronological 60/20/20 splits, and derive graph views from observed fund flows within each phase. Use six consecutive feature blocks because local features are anonymized. Use no true labels for training, calibration, or model selection.

Train three fresh cm backbones (seeds 42, 43, 44; 800 steps, 256 hidden, 3 layers, 128 targets/update, maximum 8192 training targets). Keep all nine historical score settings including the unchanged tail_flat comparator, fixed 0.5 KAN fusion and all controls. Freeze all score arrays, thresholds and hashes for all seeds before reading any labels. Evaluate known labels only with the same AP, PR-AUC, ROC-AUC, POT and q99 measures. Record graph adaptation, data and code hashes, CQ receipts, paired differences and outcome limits. Publish the reproduction and Chinese report in HIMVHAD. CQ free capacity only. Preserve other experiments and prior results.
