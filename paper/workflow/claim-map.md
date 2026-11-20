# Evidence map

| Claim | Evidence | Limit |
|---|---|---|
| Standalone zero-label package,not supervised original | model.py/network.py;docs/reproduction_CN.md | Supervised label pathways/classifier removed;not paper score replication |
| No fitting/calibration labels parsed | 5-variant absent-label tests;prediction-label guard;frozen evaluation guard | Data preparation can write labels;test labels used only in evaluation |
| Parent math preserved | results/parent_equivalence.json;dense/ego forward+gradient tests | 4 synthetic control fits;not fresh financial training |
| 18 frozen models/144 score settings/288 operating points verified | results/frozen/verification.json;per_seed.json | Saved components re-scored;CPU probes with stated tolerance |
| 48 prior configurations exact | results/frozen/historical_metric_identity.json | Historical old-data reuse is retrospective |
| One independent synthetic source evaluated | fixed source protocol,parent training manifest,CardSim results | One world,not real-bank external validation or zero-shot |
| Skill weights not stable improvement | paired_effects.csv;summary.csv;full report | No per-dataset primary substitution;3seeds not significance |
| 3 pretrained examples and wheel work | release_verification.json;pretrained provenance;CLI tests | Fixedcm/42,not best seeds;wheel excludes Git-hosted weight examples |
