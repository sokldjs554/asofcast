import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'docs/assets/live-m2'


def test_media_manifest_matches_every_retained_file():
    manifest = json.loads((MEDIA / 'media-manifest.json').read_text())
    mismatches = []
    for item in manifest['files']:
        content = (MEDIA / item['name']).read_bytes()
        if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
            mismatches.append(item['name'])
    assert not mismatches, f'Media evidence changed without manifest update: {mismatches}'


def test_media_manifest_agrees_with_capture_report_and_preserves_history():
    manifest = json.loads((MEDIA / 'media-manifest.json').read_text())
    report = json.loads((MEDIA / 'capture-report.json').read_text())
    assert manifest['ui_version'] == report['ui_version']
    assert manifest['model_run_id'] == report['model_run_id']
    assert manifest['video']['removed_leading_seconds'] == report['video_start_offset_seconds']
    assert all('provenance' in item for item in manifest['files'])
    historical = ROOT / 'docs/demo-verification-20260928.md'
    assert '2026-10-04 캡처로 갱신' in historical.read_text()
