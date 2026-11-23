# Complete — Elliptic holdout

[Task specification](prompts/005-elliptic-holdout.md) · [Chinese report](../../results/elliptic_20261124/README_CN.md) · [Verification](../../results/elliptic_20261124/verification.json)

CQ j-1124-5e4973 completed the CPU study with three 800-step backbones and 27 paired configurations. All input, code and pre-label file hashes matched. Baseline arrays were identical within each pair. Thresholds and metrics replayed on the official archive. AP 3.14% to 3.02%, POT-F1 and q99-F1 0.17% to 0%. All three proxy compactness searches met the criterion. The official endpoint returned 404 before training in the first attempt. The completed run used the verified byte-identical data mirror.

The report includes reproduction commands, comparisons with the three prior datasets, the individual seed deltas and the label coverage limitation.
