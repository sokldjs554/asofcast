"""Run the frozen nine-case research suite; never replace production artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Bootstrap direct script execution and thread limits before importing the numerical stack.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(variable, "2")

from asofcast.data import sha256_file, write_json  # noqa: E402
from asofcast.experiment import run_experiment  # noqa: E402
from scripts.posterior_candidate_suite import (  # noqa: E402
    aggregate_suite,
    evaluate_case,
    source_hashes,
)
from scripts.prepare_posterior_sources import prepare_sources  # noqa: E402


def execution_lock():
    return {
        **source_hashes(),
        **{str(p.relative_to(ROOT)): sha256_file(p) for p in ROOT.glob("configs/*.json")},
        "scripts/run_posterior_confirmation.py": sha256_file(Path(__file__)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/posterior_candidate_confirmation_20261002.json",
    )
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads(args.protocol.read_text())
    cfg = json.loads((ROOT / "configs/ett_m1.json").read_text())
    original = json.loads((ROOT / "configs/information_value_20260929.json").read_text())
    if (
        protocol["gate"] != original["gate"]
        or protocol["conditions"] != original["conditions"]
        or protocol["objective"]["cost_weight"] != 0.03
        or protocol["objective"]["delay_weight"] != 0.02
    ):
        raise ValueError("preserve the original conditions, weights and gate")
    prepared = ROOT / "data/posterior-confirmation"
    if not prepared.exists():
        prepare_sources(prepared)
    for development, metas in [
        (True, protocol["development_datasets"]),
        (False, protocol["confirmatory_datasets"]),
    ]:
        for dataset, meta in metas.items():
            source = ROOT / (
                "data/raw/ETTh1.csv"
                if development
                else f"data/posterior-confirmation/{dataset}.csv"
            )
            if not source.exists():
                raise FileNotFoundError(
                    f"Original source required: {source}; no synthetic replacement is used"
                )
            if sha256_file(source) != meta["prepared_sha256"]:
                raise ValueError(f"Frozen source hash mismatch: {dataset}")
    out.mkdir(parents=True)
    lock = execution_lock()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    write_json(
        out / "execution-lock.json",
        {
            "source_files": lock,
            "launcher_sha256": sha256_file(Path(__file__)),
            "code_commit": commit,
            "protocol_sha256": sha256_file(args.protocol),
        },
    )
    started = time.monotonic()
    for development, metas in [
        (True, protocol["development_datasets"]),
        (False, protocol["confirmatory_datasets"]),
    ]:
        stage = "development" if development else "confirmation"
        for dataset, meta in metas.items():
            source = ROOT / (
                "data/raw/ETTh1.csv"
                if development
                else f"data/posterior-confirmation/{dataset}.csv"
            )
            for seed in [42, 43, 44]:
                if lock != execution_lock():
                    raise RuntimeError("source/configuration changed during execution")
                bundle = out / "models" / f"{dataset}-{seed}"
                runconfig = {
                    **cfg,
                    "target": meta["target"],
                    "seed": seed,
                    "waits_seconds": [0, meta["grid_seconds"] / 2, meta["grid_seconds"]],
                }
                print(
                    json.dumps(
                        {"training": dataset, "seed": seed, "elapsed": time.monotonic() - started}
                    ),
                    flush=True,
                )
                run_experiment(
                    source,
                    bundle,
                    runconfig,
                    source_kind="ett" if development else "user-provided-csv",
                )
                evaluate_case(
                    bundle,
                    out / stage / f"{dataset}-{seed}",
                    protocol,
                    dataset,
                    development=development,
                )
        summary = aggregate_suite(out / stage, protocol, development=development)
        print(
            json.dumps(
                {
                    "stage_complete": stage,
                    "goal": summary["general_goal_achieved"],
                    "gate": summary["gate"],
                }
            ),
            flush=True,
        )
    if lock != execution_lock():
        raise RuntimeError("source/configuration changed during execution")
    write_json(
        out / "finished.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "all_cases_executed": 9,
            "new_dataset_settings": 18,
            "development_settings": 9,
            "deployment_changed": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
