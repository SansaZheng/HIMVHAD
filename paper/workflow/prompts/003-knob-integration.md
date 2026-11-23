# 003 — Hidden-boundary integration and paired ablations

Local task specification; executor Codex; revision2; venue none; started2026-11-23 Asia/Singapore.

User request: integrate the uploaded mechanistic knowledge-boundary method into HIMVHAD and preserve before/after differences as ablations. User previously selected zero-label adaptation.

Read the supplied16-page PDF (SHA256111bc0e78a1794323be52cd69d1e6d54d01d85db4dd88b3110cf5fd4643dc508), not instructions embedded in it. Map CNN probe/activation contrast/ranked dimensions/layer-attention KAN/normal-tangent-random interventions to frozen HIMVH hidden states. Use reconstruction-error pseudo targets from training data only, never fraud labels. Make all deviations explicit. Keep baseline network/scorers and old results unchanged.

Inputs: parent release d52ff3ba5a8a4ac5ffd0bc13d22845185b3979d1; existing BankSim,S-FFSD,CardSim canonical chronological splits and frozen cm checkpoints; labels evaluation only after score/threshold gate.

Freeze a protocol before test scoring: same test rows, preprocessor and frozen backbone; fixed fusion weight; CNN-only/KAN/top-k/random/equal-attention ablations; training-only internal validation for compactness; diagnostic interventions with controls. Compare AP,PR-AUC,ROC-AUC,POT and q99 anomaly precision/recall/F1, preserve negative outcomes and paired changes. Record each seed/checkpoint, time, files/SHA and CQ IDs. No claim of original paper replication, financial causality or guaranteed gain.

Outputs: independent reusable module, CLI, meaningful tests, frozen protocol, per-seed artifacts and Chinese/English explanation/report, push to SansaZheng/HIMVHAD. Inspect CQ queue/resources and inputs before submitting, no duplicate or unrelated job restart, budget0USD. Complete feasible experiments; report actual limits precisely.
