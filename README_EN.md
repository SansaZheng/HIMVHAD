# HIMVHAD

Independent **zero-label** financial anomaly detection adaptation of HIMVH. This is not official author code or a reproduction of the paper's supervised scores. Real training/calibration fraud labels are not used. The model learns masked-field reconstruction with native sliding-window transaction hypergraphs, cross-view discrepancies, novelty-aware message passing and modulated view fusion.

## Hidden-boundary extension (2026-11-23)

The supplied paper's CNN/latent-coordinate/KAN idea is adapted using training reconstruction-error proxy targets, without real fraud-label fitting. Nine paired frozen-backbone runs and nine scoring/ablation settings yield 81 final configurations; the initial 81-setting pilot is retained separately. Control initialization and minibatch RNG are aligned. All 35 local and CQ tests passed.

The prespecified KAN fusion changes mean AP by -10.16 pp on BankSim, +3.09 pp on S-FFSD and -0.54 pp on CardSim. POT-F1 decreases on all three. None of the nine k≤10 searches meets the 1 pp compactness criterion. These results do not establish a stable gain; the original default remains available. High symbolic fidelity (R² 0.9960–0.9985) does not imply successful fraud detection.

[Full Chinese report and per-seed tables](results/knob_20261123/README_CN.md) · [Adaptation boundaries and commands](docs/mechanistic_boundary_CN.md) · [Independent verification](results/knob_20261123/verification.json). Three fixed seed42 adapters are shipped. No original LLM/supervised paper scores or new real-bank external validation are claimed.

```bash
python -m pip install -e '.[test]'
python -m himvhad demo --output runs/demo --steps 2
python -m pytest -q
```

The demo is synthetic pipeline evidence only. For a real input/schema/checkpoint example using a synthetic financial source:

```bash
python -m himvhad fetch-benchmark --root data/benchmark --datasets cardsim
python -m himvhad predict --run examples/pretrained/cardsim \
  --data data/benchmark/cardsim --score tail --output runs/prediction
```

Three bundled pretrained examples use the fixed `cm/seed42` choice, not dataset-specific winning seeds. They are copied historical checkpoints, with scalers, calibrators, thresholds and SHA256 provenance. Downloaded dataset artifacts are pinned and hashed; their licenses are separate from the code license.

Train on the exact prior cohorts:

```bash
python -m himvhad fetch-benchmark --datasets banksim s_ffsd cardsim
python -m himvhad run --data data/benchmark/banksim --output runs/banksim \
  --variants m cm --seeds 42 43 44 --steps 800 --device cuda \
  --scores field flat tail tail_flat skill_tail skill_raw skill_equal_tail skill_hard_tail
```

Use CPU if CUDA is unavailable. Default budgets match the prior zero-label adaptation:8192 training targets,up to100000 calibration rows,128 targets/update,hidden256,3layers,window4. The full phase-local graph can include later transactions within the phase: this is offline detection,not strict causal online monitoring.

The2026-11-21 extraction release audited144 scoring configurations/288 operating points on18 frozen financial models; **zero new financial neural-network fits**.48 previous configurations match the historical metrics exactly. Retrospective BankSim/S-FFSD results are separated from one independently generated CardSim world. That synthetic-world validation is neither real-bank external validation nor zero-shot transfer. Skill weights do not consistently outperform existing scoring controls; negative results are retained. Our anomaly-class F1 is distinct from the paper's macro-F1.

[Chinese quickstart](README.md) · [Full results and limitations](results/frozen/README_CN.md) · [Method boundaries](docs/reproduction_CN.md) · [Source manifest](PORTING_MANIFEST.json) · [Release verification](results/release_verification.json).

That baseline release passed 27 local tests, an installed-wheel demo and financial audits. GitHub CI is configured and active, but no remote run was observed at release; no remote-CI pass is claimed. The extension subsequently passed 35 local and CQ tests.
