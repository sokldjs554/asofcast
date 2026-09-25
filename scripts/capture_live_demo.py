"""Exercise the real HTTP demo; keep recommendation, manual input and audit distinct."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

UI_VERSION = 'clear-flow-20260925'


def wait_for_m2(page, url: str, attempts: int = 24) -> None:
    last = None
    for _ in range(attempts):
        try:
            response = page.goto(url, wait_until='domcontentloaded', timeout=60_000)
            last = None if response is None else response.status
            if response is not None and response.status < 400:
                if (page.locator('h1').count()
                    and '센서가 늦게 도착할 때' in page.locator('h1').inner_text()
                    and page.locator(f'script[src*="{UI_VERSION}"]').count()):
                    page.wait_for_function(
                        "document.querySelectorAll('#sensorMapGrid .sensor-card').length === 7"
                        " && document.getElementById('status').textContent === ''"
                        " && !document.getElementById('runButton').disabled", timeout=60_000)
                    return
        except Exception as exc:
            if 'ERR_BLOCKED_BY_ADMINISTRATOR' in str(exc):
                raise RuntimeError('Browser policy blocks this URL; no HTTP verification was performed') from exc
            last = str(exc)
        page.wait_for_timeout(15_000)
    raise RuntimeError(f'Current public demo did not become ready: {last}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='https://asofcast.onrender.com')
    parser.add_argument('--out', type=Path, default=Path('docs/assets/live-m2'))
    parser.add_argument('--browser', help='Optional local Chromium executable')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    page_errors: list[str] = []

    with sync_playwright() as p:
        launch = {'headless': True, 'args': ['--no-sandbox']}
        if args.browser:
            launch['executable_path'] = args.browser
        browser = p.chromium.launch(**launch)
        context = browser.new_context(viewport={'width': 1440, 'height': 1000},
            device_scale_factor=1, record_video_dir=str(out),
            record_video_size={'width': 1440, 'height': 1000})
        page = context.new_page()
        page.on('pageerror', lambda exc: page_errors.append(str(exc)))

        def ready():
            page.wait_for_function("document.getElementById('status').textContent === ''"
                " && !document.getElementById('runButton').disabled", timeout=30_000)

        def top_capture(name):
            page.evaluate("window.scrollTo({top:0,behavior:'instant'})")
            page.screenshot(path=str(out / name), full_page=False)

        wait_for_m2(page, args.url)
        initial_target = page.locator('#targetTime').get_attribute('data-timestamp')
        assert initial_target, 'target must be visible before the first choice'
        initial_action = page.locator('#actionBadge').inner_text()
        initial_prediction = page.locator('#forecast').inner_text()
        assert page.locator('#technicalDetails').get_attribute('open') is None
        top_capture('m2-decision-console.png')
        page.wait_for_timeout(1800)

        # Execute exactly the displayed recommendation; never force an ACQUIRE.
        page.locator('#followRecommendation').click()
        ready()
        assert page.locator('#targetTime').get_attribute('data-timestamp') == initial_target
        assert '모델 추천' in page.locator('#resultExplanation').inner_text()
        recommendation_result = page.locator('#resultExplanation').inner_text()
        page.locator('#resultPanel').scroll_into_view_if_needed()
        page.wait_for_timeout(1800)
        page.screenshot(path=str(out / 'm2-recommended-action.png'), full_page=False)

        # Start the same case again to demonstrate an explicitly manual alternative.
        page.locator('#resetAcquisition').click()
        ready()
        assert page.locator('#forecast').inner_text() == initial_prediction
        assert page.locator('#revisionTimeline .revision-event').count() == 1
        candidate = page.locator('#sensorMapGrid .sensor-card.is-candidate').first
        sensor = candidate.locator('.sensor-card-head strong').inner_text()
        candidate.locator('.sensor-acquire').click()
        ready()
        assert page.locator('#sensorMapGrid .is-acquired').count() == 1
        assert page.locator('#beforeForecast').inner_text() == initial_prediction
        assert page.locator('#afterForecast').inner_text() == page.locator('#forecast').inner_text()
        assert page.locator('#targetTime').get_attribute('data-timestamp') == initial_target
        assert '직접 선택' in page.locator('#resultExplanation').inner_text()
        assert sensor in page.locator('#resultExplanation').inner_text()
        manual_prediction = page.locator('#forecast').inner_text()
        page.locator('#resultPanel').scroll_into_view_if_needed()
        page.wait_for_timeout(1800)
        page.screenshot(path=str(out / 'm2-after-acquisition.png'), full_page=False)

        page.locator('#technicalDetails > summary').click()
        assert page.locator('#counterfactualRows tr').count() == 7
        assert page.locator('#paretoChart circle').count() == 5
        for anchor, name in [('counterfactual', 'm2-counterfactual.png'),
                             ('revision', 'm2-revision.png'), ('pareto', 'm2-pareto.png')]:
            page.locator('#' + anchor).scroll_into_view_if_needed()
            page.wait_for_timeout(1200)
            page.screenshot(path=str(out / name), full_page=False)
        for width in [390, 320]:
            page.set_viewport_size({'width': width, 'height': 844})
            assert not page.evaluate('document.documentElement.scrollWidth > innerWidth'), f'overflow at {width}px'
        page.set_viewport_size({'width': 390, 'height': 844})
        page.locator('#technicalDetails > summary').click()
        page.locator('#resultPanel').scroll_into_view_if_needed()
        page.screenshot(path=str(out / 'm2-mobile.png'), full_page=False)
        page.set_viewport_size({'width': 1440, 'height': 1000})
        page.evaluate("window.scrollTo({top:0,behavior:'instant'})")
        page.screenshot(path=str(out / 'm2-full.png'), full_page=True)
        video = page.video
        context.close()
        browser.close()
        source_video = Path(video.path())
        target_video = out / 'asofcast-m2-demo.webm'
        if source_video.resolve() != target_video.resolve():
            shutil.move(str(source_video), target_video)

    if page_errors:
        raise RuntimeError(f'browser page errors: {page_errors}')
    names = ['m2-decision-console.png', 'm2-recommended-action.png', 'm2-after-acquisition.png',
             'm2-counterfactual.png', 'm2-revision.png', 'm2-pareto.png',
             'm2-mobile.png', 'm2-full.png', 'asofcast-m2-demo.webm']
    missing = [name for name in names if not (out / name).is_file() or not (out / name).stat().st_size]
    if missing:
        raise RuntimeError(f'missing capture outputs: {missing}')
    public = urlparse(args.url).hostname not in ('localhost', '127.0.0.1', '::1')
    report = {'status': 'captured', 'source_url': args.url, 'ui_version': UI_VERSION,
        'captured_at_unix': int(time.time()), 'browser': 'Chromium via Playwright',
        'public_http_verified': public, 'initial_action': initial_action,
        'initial_prediction': initial_prediction, 'recommendation_result': recommendation_result,
        'sensor_clicked': sensor, 'acquisition_click_mode': 'manual_after_reset',
        'manual_prediction': manual_prediction, 'fixed_target_timestamp': initial_target,
        'screenshots': [name for name in names if name.endswith('.png')],
        'video': 'asofcast-m2-demo.webm', 'scopes': [
            'real HTTP assets and model API; not the local TestClient bridge',
            'the actual recommendation is executed before an independent manual choice',
            'manual acquisition keeps the fixed target and records the real before/after values',
            'retrospective counterfactual truth remains audit-only',
            'desktop flow and 390px/320px overflow checks; not a full production load test']}
    (out / 'capture-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
