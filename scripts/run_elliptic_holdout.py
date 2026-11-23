"""Evaluate the frozen scoring protocol on the official Elliptic v1 archive."""
import argparse
import json
import resource
import time
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from himvhad.data import PHASES, boundaries
from himvhad.io import provenance, sha, write
from himvhad.knob_runner import VARIANTS, evaluate_knobs, run_knobs
from himvhad.knobs import KnobConfig
from himvhad.runner import run_group


SOURCE = "https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic-data-set?datasetVersionNumber=1"
ARCHIVE_SHA = "70ed052c52002392720615a03261039b62d1ad594f6081ad021cb174a8ad5717"
PREFIX = "elliptic_bitcoin_dataset/elliptic_txs_"
VIEWS = ["direct_flow", "shared_predecessor", "shared_successor"]
MIRROR_SHA = {
    "features": "d33d62159e64b5e889f1a7ea880227c612775b58d409598855e0c4400fa52b3e",
    "edgelist": "a2f9f6b67a39da2d8cf87fe77b9db89571ba6d880e5dd5b5991dc45c80fa34ec",
    "classes": "4ca957f0ceffd5dd164e255c7d5ad9ee69a6fa64ae1dd94d6f113e5ebf3b07ba",
}
# The uncompressed CSVs from the mirror and official release matched byte for byte.
CSV_SHA = {
    "features": "fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0",
    "edgelist": "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e",
    "classes": "93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493",
}


@contextmanager
def source_member(location, kind):
    if location.is_dir():
        archive = location / f"pyg-{kind}.zip"
        if sha(archive) != MIRROR_SHA[kind]:
            raise ValueError(f"Elliptic mirror archive SHA256 mismatch for {kind}")
        member = f"elliptic_txs_{kind}.csv"
    else:
        archive = location
        member = PREFIX + kind + ".csv"
    with ZipFile(archive) as source, source.open(member) as file:
        yield file


def sibling_pairs(flow, group_column):
    """A chain of siblings by transaction-ID order bounds edge count."""
    ordered = np.unique(flow[:, [group_column, 1 - group_column]], axis=0)
    adjacent = ordered[1:, 0] == ordered[:-1, 0]
    return np.column_stack((ordered[:-1, 1][adjacent], ordered[1:, 1][adjacent]))


def prepare(archive, output):
    if not archive.is_dir() and sha(archive) != ARCHIVE_SHA:
        raise ValueError("Official Elliptic archive SHA256 mismatch")
    output.mkdir(parents=True, exist_ok=False)
    with source_member(archive, "features") as file:
        # Col 0 is transaction ID, col 1 is the time feature. Skip 72
        # supplied one-hop aggregate features.
        frame = pd.read_csv(file, header=None, usecols=range(95),
                            dtype={i: "int64" if i < 2 else "float32" for i in range(95)})
    with source_member(archive, "edgelist") as file:
        flow = pd.read_csv(file, usecols=["txId1", "txId2"])[["txId1", "txId2"]].to_numpy()

    if len(frame) != 203769 or len(flow) != 234355 or frame[0].duplicated().any():
        raise ValueError("Unexpected Elliptic dimensions or duplicate transaction IDs")
    frame = frame.sort_values([1, 0], kind="stable")
    txids = pd.Index(frame[0])
    times = frame[1].to_numpy()
    features = frame.iloc[:, 2:].to_numpy()
    if not np.isfinite(features).all() or len(np.unique(times)) != 49:
        raise ValueError("Invalid Elliptic feature values or snapshot count")
    flow = txids.get_indexer(flow.ravel()).reshape(-1, 2)
    if (flow < 0).any() or np.any(times[flow[:, 0]] != times[flow[:, 1]]):
        raise ValueError("Elliptic edges reference missing nodes or cross snapshots")

    breaks = [int(value) for value in boundaries(times)]
    phase_ids = np.searchsorted(breaks, times, side="left")
    schema = dict(
        dataset="elliptic", features=[f"local_{i:02d}" for i in range(1, 94)],
        field_groups=[group.tolist() for group in np.array_split(np.arange(93), 6)],
        view_columns=VIEWS, view_encoding="explicit_edges", window=2,
        time_boundaries=breaks, raw_sha256=ARCHIVE_SHA, source_url=SOURCE,
        source_csv_sha256=CSV_SHA,
        protocol="whole-snapshot chronological 60/20/20; phase-local offline graph",
        splits={},
    )
    for i, phase in enumerate(PHASES):
        rows = np.flatnonzero(phase_ids == i)
        edges = flow[phase_ids[flow[:, 0]] == i] - rows[0]
        views = [edges, sibling_pairs(edges, 0), sibling_pairs(edges, 1)]
        views = [np.unique(np.sort(view, axis=1), axis=0) for view in views]
        schema["splits"][phase] = dict(rows=len(rows), time_min=int(times[rows[0]]),
                                      time_max=int(times[rows[-1]]),
                                      edges=dict(zip(VIEWS, map(len, views))))
        features[rows].astype("<f4", copy=False).tofile(output / f"{phase}.x.f32")
        np.savez_compressed(output / f"{phase}.keys.npz", **{f"k{j}": v for j, v in enumerate(views)})
    write(output / "schema.json", schema)
    return txids[phase_ids == 2]


def open_test_labels(archive, data, test_ids):
    with source_member(archive, "classes") as file:
        labels = pd.read_csv(file, dtype={"class": str}).set_index("txId")["class"]
    target = labels.reindex(test_ids).map({"1": 1, "2": 0, "unknown": -1})
    if target.isna().any():
        raise ValueError("Missing or unrecognized Elliptic labels")
    target.to_numpy(dtype="i1").tofile(data / "test.y.i8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Official ZIP or directory of verified PyG ZIPs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    data = out.parent / "prepared_elliptic"
    test_ids = prepare(args.archive, data)
    schema = json.loads((data / "schema.json").read_text())
    write(out / "data_manifest.json", schema)
    started = time.monotonic()
    seeds = [42] if args.smoke else [42, 43, 44]
    steps = 2 if args.smoke else 800
    runs = []
    for seed in seeds:
        elapsed = time.monotonic()
        assert not (data / "test.y.i8").exists()
        parent = out / "backbone" / str(seed)
        run = out / "runs" / str(seed)
        run_group(data, parent, seed=seed, steps=steps, scores=("tail_flat",), device=args.device)
        config = KnobConfig(probe_steps=2, student_steps=2) if args.smoke else KnobConfig()
        run_knobs(parent, data, run, config=config, device=args.device, evaluate=False)
        with np.load(parent / "tail_flat/scores.npz") as baseline, np.load(run / "baseline/scores.npz") as scored:
            for phase in ("calibration", "test"):
                np.testing.assert_array_equal(baseline[phase], scored[phase])
        runs.append(run)
        print(f"Frozen seed {seed}, {len(runs)}/{len(seeds)} after {time.monotonic() - elapsed:.1f} seconds", flush=True)
    write(out / "global_label_gate.json", dict(
        status="all_groups_frozen_before_any_test_label_read",
        gates={str(run.relative_to(out)): sha(run / "label_access_gate.json") for run in runs},
        baseline_equal_per_seed=True, archive_sha256=ARCHIVE_SHA,
    ))
    records = []
    if not args.smoke:
        open_test_labels(args.archive, data, test_ids)
        for run in runs:
            records.extend(evaluate_knobs(run, data))
        if len(records) != len(seeds) * len(VARIANTS):
            raise ValueError("Incomplete holdout results")
        write(out / "records.json", records)
    write(out / "complete.json", dict(
        status="passed" if args.smoke else "complete", configurations=len(records),
        steps=steps, seeds=seeds, new_backbone_fits=len(seeds),
        backbone_and_scoring_seconds=time.monotonic() - started,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        device=args.device, gpu=torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        data_transport="verified_pyg_mirror" if args.archive.is_dir() else "official_zip",
        source=provenance(), experiment_script_sha256=sha(__file__),
    ))


if __name__ == "__main__":
    main()
