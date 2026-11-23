# 004 — Paired RNG control verification and final rerun

Local task specification; executor Codex andCQ; configurationrevision2; venue none;2026-11-23 Asia/Singapore.

Static code review found control heads used a different random stream from selectedKAN,confounding ablations. Align seeds and minibatch draws to seed+1000+selected_k; do not change data,teacher,primaryscore,architecture,trainingbudget or thresholds. Code pinned2a8fb7a03344640daf43a99fc823f9f074317f6b;35local tests pass including exact initial-parameter/RNG equivalence. No new financial test metrics have been inspected.

Keep original3a74553 fulljob j-1123-21f2ed as a pilot; do not tune using its outcomes. Run35-test CQenvironment gate,then corrected81-configCPUbatch,also depends_on passedactual-shape j-1123-e2b51d. Native feature extraction and all network architectures AST-identical to earlier qualification; reuse recorded2.989GiBRAM shape proof. Budget0USD,4CPUs/8GiBRAM,2h fullbatch ceiling. Every9-group scoring artifact freezes before target parsing.

After collection independently audit thresholds,metrics,baseline identity and runtime sourcehashes. Preserve pilot separately,report RNG alignment effects without relabelling pilot as strict ablation. Publish aligned results,negative outcomes,threefixedseed42 adapters,Chinese/English explanation andsourceprovenance toHIMVHAD main.
