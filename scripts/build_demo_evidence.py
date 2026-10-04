"""Build the demo's archived research summary; refuse stale or contradictory evidence.

This never trains a model or changes the serving bundle. The original numerical
gate is reused so a positive mean cannot be mistaken for a passed condition.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from asofcast.research_diagnosis import confirmatory_gate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/research-20261002"
OUTPUT = ROOT / "src/asofcast/static/research-evidence.json"
# Verified against portable-final-suite/confirmation/aggregate.json in the
# preserved evaluation-evidence.zip. This is an immutable dated research source.
ARCHIVED_AGGREGATE_SHA256 = "4c5fbc60be75fb081c2c33ee1004ab4cf40cea2a242e2f261af9c5c7e88ae2a4"
CONDITIONS = [
    f"{dataset}/{condition}"
    for dataset in ("Taylor", "MSFT")
    for condition in ("mixed_2811", "mixed_3811", "outage_2811")
]


def build_evidence(aggregate, evaluation_bytes, decision):
    aggregate_hash = hashlib.sha256(
        json.dumps(aggregate, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if aggregate_hash != ARCHIVED_AGGREGATE_SHA256:
        raise ValueError("archived aggregate hash mismatch")
    if hashlib.sha256(evaluation_bytes).hexdigest() != decision["evaluation_sha256"]:
        raise ValueError("evaluation hash mismatch")
    evaluation = json.loads(evaluation_bytes)
    cells = aggregate["cells"]
    if set(cells) != set(CONDITIONS) or aggregate.get("development_only") is not False:
        raise ValueError("complete confirmatory conditions required")
    gate = confirmatory_gate(cells, CONDITIONS)
    # Compare unordered reasons: display order differs from the original archive.
    for recorded in (aggregate["gate"], evaluation["gate"]):
        if recorded["passed"] != gate["passed"] or set(recorded["reasons"]) != set(gate["reasons"]):
            raise ValueError("contradictory numerical gate")
    if (
        decision["gate_passed"] != gate["passed"]
        or decision["status"] != ("promoted" if gate["passed"] else "rejected")
        or decision["candidate_id"] != evaluation["candidate_id"]
        or set(decision["reasons"]) != set(gate["reasons"])
        or aggregate["general_goal_achieved"] != gate["passed"]
        or evaluation["general_goal_achieved"] != gate["passed"]
    ):
        raise ValueError("contradictory release decision")
    rows = []
    for condition in CONDITIONS:
        record = cells[condition]["comparisons"]["validation_simple"]
        values = [record[key] for key in ("relative_improvement", "lower95", "upper95")]
        if not all(math.isfinite(value) for value in values) or values[1] > values[2]:
            raise ValueError("finite ordered research interval required")
        rows.append(
            {
                "condition": condition,
                "improvement_percent": values[0] * 100,
                "lower95": values[1],
                "upper95": values[2],
                "improved_seeds": sum(value > 0 for value in record["seed_improvements"]),
                "passed": confirmatory_gate(cells, [condition])["passed"],
            }
        )
    return {
        "schema": "asofcast.demo-research-evidence.v1",
        "research_date": "2026-10-02",
        "candidate_id": evaluation["candidate_id"],
        "status": decision["status"],
        "gate_passed": gate["passed"],
        "evaluation_sha256": decision["evaluation_sha256"],
        "aggregate_sha256": aggregate_hash,
        "total_conditions": len(rows),
        "passed_conditions": sum(row["passed"] for row in rows),
        "training_seeds": 3,
        "rows": rows,
        "scope": aggregate["scope"],
    }


def encode(data):
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def check_output(path, data):
    if not path.is_file() or path.read_text(encoding="utf-8") != encode(data):
        raise ValueError("stale demo evidence: run scripts/build_demo_evidence.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = build_evidence(
        json.loads((SOURCE / "confirmation-aggregate.json").read_text(encoding="utf-8")),
        (SOURCE / "promotion-summary.json").read_bytes(),
        json.loads((SOURCE / "release-decision.json").read_text(encoding="utf-8")),
    )
    if args.check:
        check_output(OUTPUT, data)
    else:
        OUTPUT.write_text(encode(data), encoding="utf-8")


if __name__ == "__main__":
    main()
