import copy
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from scripts.build_demo_evidence import build_evidence, check_output

ROOT = Path(__file__).resolve().parents[1]


def sources():
    folder = ROOT / "docs/research-20261002"
    return (
        json.loads((folder / "confirmation-aggregate.json").read_text()),
        (folder / "promotion-summary.json").read_bytes(),
        json.loads((folder / "release-decision.json").read_text()),
    )


def test_complete_results_keep_positive_mean_failure_and_rejection():
    data = build_evidence(*sources())
    assert data["passed_conditions"] == 1
    assert data["total_conditions"] == 6
    assert data["status"] == "rejected"
    outage = next(row for row in data["rows"] if row["condition"] == "Taylor/outage_2811")
    assert outage["improvement_percent"] == pytest.approx(5.597992125358117)
    assert outage["lower95"] < 0 < outage["upper95"]
    assert outage["passed"] is False


@pytest.mark.parametrize("mutation", ["missing", "nan", "contradictory"])
def test_incomplete_or_contradictory_research_cannot_be_published(mutation):
    aggregate, evaluation, decision = sources()
    aggregate = copy.deepcopy(aggregate)
    if mutation == "missing":
        del aggregate["cells"]["MSFT/outage_2811"]
    elif mutation == "nan":
        aggregate["cells"]["Taylor/outage_2811"]["comparisons"]["validation_simple"]["upper95"] = (
            float("nan")
        )
    else:
        decision["status"] = "accepted"
    with pytest.raises(ValueError):
        build_evidence(aggregate, evaluation, decision)


def test_evaluation_hash_binds_displayed_release_decision():
    aggregate, evaluation, decision = sources()
    with pytest.raises(ValueError, match="hash"):
        build_evidence(aggregate, evaluation + b" ", decision)


def test_changed_numbers_cannot_reuse_the_original_gate_and_release_hash():
    aggregate, evaluation, decision = sources()
    aggregate["cells"]["Taylor/mixed_2811"]["comparisons"]["validation_simple"][
        "relative_improvement"
    ] = 0.5
    with pytest.raises(ValueError, match="aggregate hash"):
        build_evidence(aggregate, evaluation, decision)


def test_stale_generated_output_is_rejected(tmp_path):
    output = tmp_path / "research-evidence.json"
    output.write_text("{}")
    with pytest.raises(ValueError, match="stale"):
        check_output(output, build_evidence(*sources()))


def test_packaged_research_evidence_matches_preserved_sources():
    from scripts import build_demo_evidence as builder

    assert hasattr(builder, "build_latest_evidence")
    source = (ROOT / "docs/research-20261005/confirmation-summary.json").read_bytes()
    data = builder.build_latest_evidence(source)
    assert data["research_date"] == "2026-10-05"
    assert data["passed_conditions"] == 0
    assert data["status"] == "rejected"
    seoul = next(r for r in data["rows"] if r["condition"] == "SeoulBike/outage_2811")
    assert seoul["improvement_percent"] == pytest.approx(-0.4299461546692029)
    assert seoul["upper95"] < 0
    check_output(ROOT / "src/asofcast/static/research-evidence.json", data)


def test_latest_research_rejects_modified_source_bytes():
    from scripts import build_demo_evidence as builder

    assert hasattr(builder, "build_latest_evidence")
    source = (ROOT / "docs/research-20261005/confirmation-summary.json").read_bytes()
    with pytest.raises(ValueError, match="hash"):
        builder.build_latest_evidence(source + b" ")


def test_demo_repository_evidence_links_resolve_to_tracked_files():
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.urls = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.urls.append(dict(attrs).get("href", ""))

    parser = Links()
    parser.feed((ROOT / "src/asofcast/static/index.html").read_text())
    prefix = "/sokldjs554/asofcast/blob/main/"
    for url in parser.urls:
        parsed = urlparse(url)
        if parsed.netloc == "github.com" and parsed.path.startswith(prefix):
            assert (ROOT / unquote(parsed.path[len(prefix) :])).is_file(), url
