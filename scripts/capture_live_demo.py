"""Record a short real-HTTP walkthrough, then capture separate audit screenshots."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

UI_VERSION = 'submission-20261005-r2'
VIEWPORT = {'width': 1440, 'height': 1000}


def wait_for_m2(page, url: str, attempts: int = 24) -> None:
    last = None
    for attempt in range(1, attempts + 1):
        try:
            # A repeated goto can reuse a fresh cached pre-deploy HTML document.
            # Reload revalidates that document while Render finishes updating.
            if attempt > 1 and urlparse(page.url).netloc == urlparse(url).netloc:
                response = page.reload(wait_until='domcontentloaded', timeout=60_000)
            else:
                response = page.goto(url, wait_until='domcontentloaded', timeout=60_000)
            last = None if response is None else response.status
            if response is not None and response.status < 400:
                last = {'status': response.status, 'title': page.title(),
                        'scripts': page.locator('script[src]').evaluate_all(
                            '(nodes) => nodes.map(node => node.getAttribute("src"))')}
                if (page.locator('h1').count()
                    and '센서가 늦게 도착할 때' in page.locator('h1').inner_text()
                    and page.locator(f'script[src*="{UI_VERSION}"]').count()):
                    ready(page)
                    page.wait_for_function(
                        "document.querySelectorAll('#sensorMapGrid .sensor-card').length === 7",
                        timeout=60_000)
                    print(f'Ready: {url} ({UI_VERSION})', flush=True)
                    return
        except Exception as exc:
            if 'ERR_BLOCKED_BY_ADMINISTRATOR' in str(exc):
                raise RuntimeError('Browser policy blocks this URL; no HTTP verification was performed') from exc
            last = str(exc)
        print(f'Waiting for current demo ({attempt}/{attempts}): {last}', flush=True)
        page.wait_for_timeout(15_000)
    raise RuntimeError(f'Current public demo did not become ready: {last}')


def ready(page):
    page.wait_for_function("document.getElementById('status').textContent === ''"
        " && !document.getElementById('runButton').disabled", timeout=30_000)


def frame(page, anchor: str | None = None):
    page.evaluate("""(anchor) => {
        const inset = document.querySelector('header').getBoundingClientRect().height + 24;
        const top = anchor ? document.getElementById(anchor).getBoundingClientRect().top + scrollY - inset : 0;
        window.scrollTo({top, behavior:'instant'});
    }""", anchor)


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
        # Render cold starts and deployment propagation must not pad the short clip.
        warm_context = browser.new_context(viewport=VIEWPORT)
        wait_for_m2(warm_context.new_page(), args.url)
        warm_context.close()

        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1,
            record_video_dir=str(out), record_video_size=VIEWPORT)
        recording_started = time.monotonic()
        page = context.new_page()
        page.on('pageerror', lambda exc: page_errors.append(str(exc)))
        wait_for_m2(page, args.url, attempts=2)
        state = page.evaluate("async () => (await fetch('/api/acquisition?case_id=0&scenario=mixed&wait_seconds=0')).json()")
        initial_target = page.locator('#targetTime').get_attribute('data-timestamp')
        assert initial_target, 'target must be visible before the first choice'
        initial_action = page.locator('#actionBadge').inner_text()
        initial_prediction = page.locator('#forecast').inner_text()
        assert page.locator('#technicalDetails').get_attribute('open') is None
        frame(page)
        video_start_offset_seconds = round(time.monotonic() - recording_started, 3)
        page.screenshot(path=str(out / 'm2-decision-console.png'))
        page.wait_for_timeout(3000)

        # The three counterfactual branches all begin from a fresh, identical state.
        page.locator('#resetAcquisition').click()
        ready(page)
        assert page.locator('#forecast').inner_text() == initial_prediction
        assert page.locator('#revisionTimeline .revision-event').count() == 1
        sensor = page.locator('#sensorMapGrid .is-candidate .sensor-card-head strong').first.inner_text()
        page.locator('#compareSensor').select_option(sensor)
        frame(page)
        page.locator('#previewChoices').click()
        page.wait_for_function("!document.getElementById('compareChoices').disabled")
        assert page.locator('#choiceComparison [data-status="ok"]').count() == 3
        assert page.locator('#revisionTimeline .revision-event').count() == 1
        assert page.locator('#sensorMapGrid .is-acquired').count() == 0
        assert page.locator('#forecast').inner_text() == initial_prediction
        frame(page, 'comparison')
        page.screenshot(path=str(out / 'm2-choice-comparison.png'))
        page.wait_for_timeout(6000)
        page.locator('#showComparisonTruth').check()
        assert page.locator('.choice-error:visible').count() == 3
        page.screenshot(path=str(out / 'm2-choice-audit.png'))
        page.wait_for_timeout(4000)
        with page.expect_download() as download:
            page.locator('#exportComparison').click()
        download.value.save_as(out / 'comparison-example.json')
        comparison = json.loads((out / 'comparison-example.json').read_text(encoding='utf-8'))
        assert comparison['run_id'] == state['run_id']
        assert comparison['scope'] == 'single_case_counterfactual_not_executed'
        assert all(str(row['target_time']) == initial_target for row in comparison['choices'])
        page.wait_for_timeout(1000)

        page.locator('#followRecommendation').click()
        ready(page)
        assert page.locator('#targetTime').get_attribute('data-timestamp') == initial_target
        recommendation_result = page.locator('#resultExplanation').inner_text()
        assert '모델 추천' in recommendation_result
        frame(page, 'resultPanel')
        page.screenshot(path=str(out / 'm2-recommended-action.png'))
        page.wait_for_timeout(4000)

        page.locator('#resetAcquisition').click()
        ready(page)

        # A manual sensor choice is a separate, real session action.
        candidate = page.locator('#sensorMapGrid .sensor-card.is-candidate').first
        sensor = candidate.locator('.sensor-card-head strong').inner_text()
        candidate.locator('.sensor-acquire').click()
        ready(page)
        assert page.locator('#sensorMapGrid .is-acquired').count() == 1
        assert page.locator('#beforeForecast').inner_text() == initial_prediction
        assert page.locator('#afterForecast').inner_text() == page.locator('#forecast').inner_text()
        assert page.locator('#targetTime').get_attribute('data-timestamp') == initial_target
        assert '직접 선택' in page.locator('#resultExplanation').inner_text()
        assert sensor in page.locator('#resultExplanation').inner_text()
        manual_prediction = page.locator('#forecast').inner_text()
        frame(page, 'resultPanel')
        page.screenshot(path=str(out / 'm2-after-acquisition.png'))
        page.wait_for_timeout(4000)
        frame(page)
        page.wait_for_timeout(2000)
        video = page.video
        context.close()
        target_video = out / 'asofcast-m2-demo.webm'
        shutil.move(str(video.path()), target_video)

        # Longer expert and mobile checks are outside the recorded walkthrough.
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()
        page.on('pageerror', lambda exc: page_errors.append(str(exc)))
        wait_for_m2(page, args.url, attempts=2)
        page.locator('#sensorMapGrid .is-candidate .sensor-acquire').first.click()
        ready(page)
        page.locator('.engineering-evidence-head a[href="#researchEvidence"]').click()
        page.wait_for_function("document.querySelectorAll('#researchRows tr').length === 6")
        assert '승격 거절' in page.locator('#researchVerdict').inner_text()
        assert '0 / 6' in page.locator('#researchVerdict').inner_text()
        assert page.locator('.engineering-evidence-grid a').count() == 6
        frame(page, 'researchEvidence')
        page.screenshot(path=str(out / 'm2-research-evidence.png'))
        page.locator('#engineeringEvidenceTitle').scroll_into_view_if_needed()
        page.screenshot(path=str(out / 'm2-engineering-evidence.png'))
        assert page.locator('#counterfactualRows tr').count() == 7
        assert page.locator('#paretoChart circle').count() == 5
        for anchor, name in [('counterfactual', 'm2-counterfactual.png'),
                             ('revision', 'm2-revision.png'), ('pareto', 'm2-pareto.png')]:
            frame(page, anchor)
            page.screenshot(path=str(out / name))
        page.locator('#technicalDetails > summary').click()
        page.locator('#compareChoices').click()
        page.wait_for_function("!document.getElementById('compareChoices').disabled")
        for width in [390, 320]:
            page.set_viewport_size({'width': width, 'height': 844})
            page.locator('#technicalDetails').evaluate('(element) => element.open = true')
            assert not page.evaluate('document.documentElement.scrollWidth > innerWidth'), f'overflow at {width}px'
        page.locator('#technicalDetails').evaluate('(element) => element.open = false')
        page.set_viewport_size({'width': 390, 'height': 844})
        frame(page)
        page.screenshot(path=str(out / 'm2-mobile.png'))
        page.set_viewport_size(VIEWPORT)
        frame(page)
        page.screenshot(path=str(out / 'm2-full.png'), full_page=True)
        context.close()
        browser.close()

    if page_errors:
        raise RuntimeError(f'browser page errors: {page_errors}')
    names = ['m2-decision-console.png', 'm2-recommended-action.png', 'm2-after-acquisition.png',
             'm2-choice-comparison.png', 'm2-choice-audit.png', 'comparison-example.json',
             'm2-counterfactual.png', 'm2-revision.png', 'm2-pareto.png',
             'm2-mobile.png', 'm2-full.png', 'm2-research-evidence.png',
             'm2-engineering-evidence.png', 'asofcast-m2-demo.webm']
    missing = [name for name in names if not (out / name).is_file() or not (out / name).stat().st_size]
    if missing:
        raise RuntimeError(f'missing capture outputs: {missing}')
    public = urlparse(args.url).hostname not in ('localhost', '127.0.0.1', '::1')
    report = {'status': 'captured', 'source_url': args.url, 'ui_version': UI_VERSION,
        'captured_at_unix': int(time.time()), 'browser': 'Chromium via Playwright',
        'public_http_verified': public, 'model_run_id': state['run_id'],
        'source_kind': state['source_kind'], 'initial_action': initial_action,
        'initial_prediction': initial_prediction, 'recommendation_result': recommendation_result,
        'sensor_clicked': sensor, 'acquisition_click_mode': 'manual_after_reset',
        'manual_prediction': manual_prediction, 'fixed_target_timestamp': initial_target,
        'screenshots': [name for name in names if name.endswith('.png')],
        'video': 'asofcast-m2-demo.webm', 'comparison_evidence': 'comparison-example.json',
        'video_start_offset_seconds': video_start_offset_seconds,
        'research_evidence_verified': {'date': '2026-10-05', 'conditions': 6,
            'passed_conditions': 0, 'release_status': 'rejected', 'role_evidence_links': 6},
        'scopes': [
            'real HTTP assets and model API; not the local TestClient bridge',
            'cold start and expert/mobile checks excluded from short video',
            'independent alternatives and opt-in audit precede the actual recommendation and a manual choice',
            'comparison preserves session state, fixed target and opt-in retrospective truth',
            'manual acquisition records the real before/after values',
            'desktop flow and 390px/320px overflow checks; not a full production load test']}
    (out / 'capture-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
