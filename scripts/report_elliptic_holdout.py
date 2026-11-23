"""Verify and summarize a completed Elliptic CQ run."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from himvhad.evaluation import metrics
from himvhad.io import sha, write
from himvhad.knob_runner import VARIANTS
from himvhad.thresholds import calibrate
from scripts.run_elliptic_holdout import ARCHIVE_SHA, open_test_labels, prepare


def read_json(path):
    return json.loads(path.read_text())


def verify_run(run, parent, gate_hash, labels):
    gate_file = run / "label_access_gate.json"
    assert sha(gate_file) == gate_hash
    for name, expected in read_json(gate_file)["files_sha256"].items():
        assert sha(run / name) == expected, name
    results = []
    for variant in VARIANTS:
        variant_dir = run / variant
        with np.load(variant_dir / "scores.npz", allow_pickle=False) as scores:
            calibration, test = scores["calibration"], scores["test"]
        if variant == "baseline":
            with np.load(parent / "tail_flat/scores.npz", allow_pickle=False) as before:
                np.testing.assert_array_equal(before["test"], test)
                np.testing.assert_array_equal(before["calibration"], calibration)
        thresholds = read_json(variant_dir / "thresholds.json")
        for name, expected in calibrate(calibration).items():
            assert thresholds[name]["status"] == expected["status"], name
            if expected["status"] == "ok":
                np.testing.assert_allclose(thresholds[name]["threshold"], expected["threshold"],
                                           rtol=1e-10, atol=1e-10)
        record = read_json(variant_dir / "result.json")
        assert record["gate_sha256"] == gate_hash
        assert record["scores_sha256"] == sha(variant_dir / "scores.npz")
        recomputed = metrics(labels, test, thresholds)
        for name, value in recomputed.items():
            if name == "operating_points":
                assert record["metrics"][name] == value, (variant, name)
            else:
                np.testing.assert_allclose(record["metrics"][name], value, rtol=1e-12, atol=1e-12)
        results.append(record)
    return results


def write_metrics(output, records):
    rows = []
    for record in records:
        result = dict(seed=record["seed"], variant=record["variant"])
        result.update({key: value for key, value in record["metrics"].items() if key != "operating_points"})
        for threshold, point in record["metrics"]["operating_points"].items():
            for key in ["precision", "recall", "f1", "fpr", "alarm_fraction", "tp", "fp", "tn", "fn"]:
                result[f"{threshold}_{key}"] = point[key] if point is not None else None
        rows.append(result)

    def table(path, data):
        with (output / path).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)

    table("metrics.csv", rows)
    baseline = {row["seed"]: row for row in rows if row["variant"] == "baseline"}
    paired = []
    for row in rows:
        base = baseline[row["seed"]]
        differences = {}
        for name in ("average_precision", "pr_auc_trapezoid", "roc_auc", "pot_f1", "quantile_f1"):
            value, ref = row[name], base[name]
            differences[f"{name}_delta_pp"] = 100 * (value - ref) if value is not None and ref is not None else None
        paired.append(dict(seed=row["seed"], variant=row["variant"], **differences))
    table("paired_differences.csv", paired)
    return rows, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    experiment, output = args.experiment, args.report
    complete = read_json(experiment / "complete.json")
    assert (complete["status"], complete["configurations"], complete["steps"]) == ("complete", 27, 800)
    source_hashes = {f"himvhad/{name}": value for name, value in complete["source"]["source_sha256"].items()}
    source_hashes["scripts/run_elliptic_holdout.py"] = complete["experiment_script_sha256"]
    for name, expected in source_hashes.items():
        committed = subprocess.check_output(["git", "show", f"{args.commit}:{name}"])
        assert hashlib.sha256(committed).hexdigest() == expected, name
    gate = read_json(experiment / "global_label_gate.json")
    assert gate["archive_sha256"] == ARCHIVE_SHA
    assert gate["status"] == "all_groups_frozen_before_any_test_label_read"
    assert set(gate["gates"]) == {"runs/42", "runs/43", "runs/44"}
    test_ids = prepare(args.archive, args.scratch)
    schema = read_json(args.scratch / "schema.json")
    assert schema == read_json(experiment / "data_manifest.json")
    open_test_labels(args.archive, args.scratch, test_ids)
    labels_path = args.scratch / "test.y.i8"
    labels = np.fromfile(labels_path, dtype="i1")
    records, adapters = [], {}
    for seed in (42, 43, 44):
        run, parent = experiment / "runs" / str(seed), experiment / "backbone" / str(seed)
        manifest = read_json(run / "run_manifest.json")
        assert manifest["seed"] == seed
        for name, expected in manifest["input_sha256"].items():
            assert sha(args.scratch / name) == expected, name
        for name, expected in manifest["parent_sha256"].items():
            assert sha(parent / name) == expected, name
        results = verify_run(run, parent, gate["gates"][f"runs/{seed}"], labels)
        assert all(record["target_sha256"] == sha(labels_path) for record in results)
        records.extend(results)
        info = read_json(run / "adapter_info.json")
        adapters[str(seed)] = {key: info[key] for key in (
            "selected_k", "selected_dimensions", "near_optimal_proxy_criterion_met",
            "teacher_validation_balanced_accuracy", "sweep", "symbolic_validation_r2",
            "controls_validation_balanced_accuracy", "selected_student_and_controls_rng_seed")}
    assert records == read_json(experiment / "records.json")
    output.mkdir(parents=True, exist_ok=False)
    rows, paired = write_metrics(output, records)
    write(output / "per_seed.json", records)
    write(output / "adapter_diagnostics.json", adapters)
    write(output / "data_manifest.json", schema)
    write(output / "verification.json", dict(
        status="passed", code_commit=args.commit, score_configurations=27,
        archived_metrics_replayed=True, baselines_exact=True, source_hashes_checked=len(source_hashes),
        test_target_sha256=sha(labels_path), execution=complete, label_gate=gate,
    ))

    def display(samples):
        values = np.array([value for value in samples if value is not None])
        if len(values) < 2:
            return f"{values[0]:.2f}（1/3）" if len(values) else "不可用"
        return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}" + ("（2/3）" if len(values) == 2 else "")

    columns = ["average_precision", "pr_auc_trapezoid", "roc_auc", "pot_f1", "quantile_f1"]
    lines = [
        "# Elliptic 独立数据验证", "",
        "使用 Elliptic v1 的真实比特币交易记录。训练、校准和测试分别包含时间步 1–30、31–41、42–49。",
        "模型使用 93 个交易本地特征，时间用于切分。匿名特征按原列顺序分为六组。"
        "三个图视图来自直接资金流、共同上游和共同下游。后两个视图按交易 ID 排序连接相邻项，"
        "边仅来自同一阶段内的交易快照。模型基于完整快照进行离线检测。", "",
        f"训练 123,287 笔，校准 43,658 笔，测试 36,824 笔。"
        f"已知测试标签 {np.sum(labels >= 0):,} 笔，其中非法交易 {np.sum(labels == 1):,} 笔。"
        f"其余 {np.sum(labels < 0):,} 笔参与图上下文与打分，指标只统计已知标签。", "",
        "[官方数据](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) · "
        "[数据论文](https://arxiv.org/abs/1908.02591) · [冻结协议](../../paper/workflow/prompts/005-elliptic-holdout.md)", "",
        "首个 CQ 作业访问官方链接时收到 404，未开始训练。完成实验的作业使用 PyTorch Geometric 镜像，"
        "三个 CSV 的解压后 SHA256 与官方 v1 归档逐一相等。复现可使用官方归档或经过校验的镜像。",
        "[PyG 下载代码](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/elliptic.html)", "",
        "三个新骨干种子各训练 800 步，每个种子的融合与消融共用该种子的骨干。"
        "CNN/KAN 只学习训练数据的重构误差代理目标，主融合权重固定为 0.5。"
        "三个种子的所有分数和阈值保存后才打开标签文件。下表为种子均值 ± 样本标准差，单位为百分比。", "",
        "| 评分 | AP | 梯形 PR-AUC | ROC-AUC | POT-F1 | q99-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        group = [row for row in rows if row["variant"] == variant]
        values = [display([100 * row[key] if row[key] is not None else None for row in group]) for key in columns]
        lines.append(f"| {variant} | " + " | ".join(values) + " |")
    fusion = [row for row in paired if row["variant"] == "kan_fused"]
    baseline_rows = [row for row in rows if row["variant"] == "baseline"]
    fusion_rows = [row for row in rows if row["variant"] == "kan_fused"]
    def mean(group, key):
        return np.mean([row[key] for row in group])

    delta_ap = mean(fusion, "average_precision_delta_pp")
    wins = sum(row["average_precision_delta_pp"] > 0 for row in fusion)
    highest_ap = max(VARIANTS, key=lambda v: mean([row for row in rows if row["variant"] == v], "average_precision"))
    lines += [
        "", f"主融合的平均 AP 变化为 {delta_ap:+.2f} 个百分点，{wins}/3 个种子的 AP 高于原版。"
        f"全部设置中平均 AP 最高的是 `{highest_ap}`。已知测试标签中非法交易占 "
        f"{100 * (labels == 1).sum() / (labels >= 0).sum():.2f}%，对应随机排序的 AP 参考水平。",
        "", "| 评分 | POT 精确率 | POT 召回率 | q99 精确率 | q99 召回率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, group in [("baseline", baseline_rows), ("kan_fused", fusion_rows)]:
        values = [display([100 * row[key] if row[key] is not None else None for row in group])
                  for key in ("pot_precision", "pot_recall", "quantile_precision", "quantile_recall")]
        lines.append(f"| {variant} | " + " | ".join(values) + " |")
    lines += ["", "融合减原版的配对变化（百分点）", "", "| 种子 | ΔAP | ΔPOT-F1 | Δq99-F1 |",
              "|---|---:|---:|---:|"]
    for row in fusion:
        values = [row[name + "_delta_pp"] for name in ("average_precision", "pot_f1", "quantile_f1")]
        lines.append(f"| {row['seed']} | " + " | ".join(f"{v:+.2f}" if v is not None else "不可用" for v in values) + " |")
    previous = Path(__file__).resolve().parents[1] / "results/knob_20261123/summary.csv"
    with previous.open() as file:
        history = {(row["dataset"], row["variant"]): row for row in csv.DictReader(file)}
    lines += ["", "与前三个数据集一起看融合方向，差值都以各自的原版为基准（百分点）。", "",
              "| 数据集 | ΔAP | ΔPOT-F1 | Δq99-F1 |", "|---|---:|---:|---:|"]
    for key, label in [("banksim", "BankSim"), ("s_ffsd", "S-FFSD"), ("cardsim", "CardSim")]:
        before, after = (history[key, variant] for variant in ("baseline", "kan_fused"))
        values = [100 * (float(after[name]) - float(before[name]))
                  for name in ("average_precision_mean", "pot_f1_mean", "quantile_f1_mean")]
        lines.append(f"| {label} | " + " | ".join(f"{v:+.2f}" for v in values) + " |")
    columns = ["average_precision_delta_pp", "pot_f1_delta_pp", "quantile_f1_delta_pp"]
    complete_deltas = [mean(fusion, col) if all(row[col] is not None for row in fusion)
                       else None for col in columns]
    lines.append("| Elliptic | " + " | ".join(f"{v:+.2f}" if v is not None else "不可用" for v in complete_deltas) + " |")
    reached = sum(info["near_optimal_proxy_criterion_met"] for info in adapters.values())
    lines += ["", f"内部代理验证有 {reached}/3 组达到 10 维以内、距全维 CNN 平衡准确率至多 1 个百分点的标准。",
              f"正式运行使用 {complete['device']}，三个种子的拟合和评分共用时 "
              f"{complete['backbone_and_scoring_seconds']/60:.1f} 分钟。原基线逐点一致，27 组分数和阈值通过独立回放。", "",
              "[逐种子指标](metrics.csv) · [配对差值](paired_differences.csv) · "
              "[压缩诊断](adapter_diagnostics.json) · [核验记录](verification.json)", "",
              "各数据集的输入特征、图结构和标签覆盖不同。Elliptic 的结果适用于这组公开比特币快照的已知标签样本。"
              "三组配对结果反映初始化差异，跨数据来源仍需结合此前三个数据集的结果判断。", "",
              "复现实验", "", "```bash",
              "python -m pip install -e '.[test]'",
              "curl -fL 'https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic-data-set?datasetVersionNumber=1' -o elliptic.zip",
              "PYTHONPATH=. python scripts/run_elliptic_holdout.py --archive elliptic.zip --output runs/elliptic --device cpu",
              "```", "",
              "English: This experiment evaluates the prespecified zero-label fusion on temporally held-out Elliptic "
              "Bitcoin transactions using three matched backbones and the same nine scoring settings.", ""]
    (output / "README_CN.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
