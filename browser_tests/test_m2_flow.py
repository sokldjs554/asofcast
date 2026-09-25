"""Complete current M2 flow; optional screenshots are always labelled local/ASGI."""
from __future__ import annotations

import json
import os
from pathlib import Path


def test_summary_does_not_claim_wrong_source_or_cloud_deployment(page):
    meta = page.evaluate("async () => JSON.parse((await asgiFetch('/api/metadata', 'GET', null)).body)")
    summary = page.locator('.verification-grid article').nth(0).locator('strong').inner_text()
    if meta['source_kind'] == 'ett':
        assert 'ETTh1' in summary and '합성' not in summary
    elif meta['source_kind'] == 'synthetic':
        assert '합성' in summary
    if not meta['cloud_deployed']:
        assert 'Render · LIVE' not in page.locator('.verification-grid').inner_text()


def test_complete_m2_desktop_mobile_flow(page):
    out = os.environ.get('ASOFCAST_CAPTURE_DIR')
    def capture(name):
        if out:
            folder = Path(out); folder.mkdir(parents=True, exist_ok=True)
            page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
            page.wait_for_function('window.scrollY === 0')
            page.screenshot(path=str(folder / (name + '.png')), full_page=True)
    def ready():
        page.wait_for_function("document.getElementById('status').textContent === '' && !document.getElementById('runButton').disabled")
    assert page.locator('h1').inner_text() == 'AI Decision Console'
    assert page.locator('#sensorMapGrid .sensor-card').count() == 7
    assert page.locator('#counterfactualRows tr').count() == 7
    assert page.locator('#paretoChart circle').count() == 5
    expected = page.evaluate("""async () => {
        const r = JSON.parse((await asgiFetch('/api/acquisition?case_id=0&scenario=mixed&wait_seconds=0', 'GET', null)).body);
        return r.prediction.toLocaleString('ko-KR', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }""")
    assert page.locator('#forecast').inner_text() == expected
    capture('m2-desktop-before')
    page.locator('#sensorMapGrid .is-candidate .sensor-acquire').first.click()
    ready()
    assert page.locator('#sensorMapGrid .is-acquired').count() == 1
    assert '취득 ·' in page.locator('#revisionTimeline').inner_text()
    capture('m2-desktop-acquired')
    page.locator('#resetAcquisition').click()
    ready()
    assert page.locator('#sensorMapGrid .is-acquired').count() == 0
    page.locator('#scenario').select_option('outage')
    ready()
    page.locator('#waitSelect').select_option('1800')
    ready()
    page.locator('#nextCase').click()
    ready()
    assert page.locator('#caseId').input_value() == '1'
    page.set_viewport_size({'width': 390, 'height': 844})
    assert not page.evaluate('document.documentElement.scrollWidth > window.innerWidth'), 'mobile document overflow'
    capture('m2-mobile-outage')
    page.locator('#resetAcquisition').click()
    ready()
    # Scroll does not change the model or action state.
    page.locator('#counterfactual').scroll_into_view_if_needed()
    assert page.locator('#counterfactualRows tr').count() == 7
    if out:
        Path(out, 'transport.json').write_text(json.dumps({
            'transport': 'local Chromium DOM + real FastAPI/PyTorch TestClient bridge',
            'public_http_verified': False,
            'requests': page.evaluate('window.traffic'),
        }, ensure_ascii=False, indent=2), encoding='utf-8')
