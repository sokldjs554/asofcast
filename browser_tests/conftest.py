"""Real model/API responses, with browser-local response-delivery fault injection."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def model_dir(tmp_path_factory):
    explicit = os.environ.get('ASOFCAST_TEST_ARTIFACTS')
    if explicit:
        return Path(explicit)
    from asofcast.data import write_synthetic_csv
    from asofcast.experiment import run_experiment
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp('browser-model')
    source = root / 'source.csv'
    write_synthetic_csv(source, n=720, seed=9)
    out = root / 'run'
    run_experiment(source, out, {'lookback': 12, 'epochs': 2, 'policy_epochs': 3,
                                'acquisition_epochs': 3}, source_kind='synthetic')
    return out


@pytest.fixture(scope='session')
def browser():
    with sync_playwright() as p:
        kwargs = {'headless': True, 'args': ['--no-sandbox']}
        executable = os.environ.get('ASOFCAST_BROWSER')
        if executable:
            kwargs['executable_path'] = executable
        browser = p.chromium.launch(**kwargs)
        yield browser
        browser.close()


@pytest.fixture
def page(browser, model_dir):
    from asofcast.service import create_app
    with TestClient(create_app(model_dir)) as client:
        context = browser.new_context(viewport={'width': 1440, 'height': 1000})
        page = context.new_page()
        page.set_default_timeout(7000)
        errors = []
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        def bridge(_source, url, method, body):
            response = client.request(method, url, content=body,
                                      headers={'Content-Type': 'application/json'})
            return {'status': response.status_code, 'body': response.text}
        page.expose_binding('asgiFetch', bridge)
        html = client.get('/').text
        html = re.sub(r'<script\b[^>]*>.*?</script>', '', html, flags=re.S)
        html = re.sub(r'<link\b[^>]*>', '', html)
        page.set_content(html)
        page.add_style_tag(content=client.get('/static/style.css').text)
        page.evaluate("""() => {
          window.traffic = []; window.holdNext = null; window.held = null; window.failNext = null;
          window.fetch = async (url, options = {}) => {
            const u = new URL(String(url), 'http://asofcast.test');
            const call = {path: u.pathname, caseId: u.searchParams.get('case_id'),
                          method: options.method || 'GET', body: options.body || null};
            window.traffic.push(call);
            const hold = window.holdNext;
            const matched = hold && hold.path === call.path
              && (hold.caseId === undefined || hold.caseId === call.caseId);
            if (matched) window.holdNext = null;
            const r = await window.asgiFetch(u.pathname + u.search, call.method, call.body);
            if (window.failNext === call.path) {
              window.failNext = null;
              r.status = 503; r.body = JSON.stringify({detail: 'injected transport failure'});
            }
            call.status = r.status;
            if (matched) {
              await new Promise(resolve => { window.held = {release: resolve, call}; });
              window.held = null;
            }
            // Deliberately deliver even after AbortController aborts: cancellation
            // is best effort, so the application must also reject stale responses.
            call.delivered = true;
            return new Response(r.body, {status: r.status,
              headers: {'Content-Type': 'application/json'}});
          };
        }""")
        page.add_script_tag(content=client.get('/static/app.js').text)
        page.wait_for_function("document.querySelectorAll('#sensorMapGrid .sensor-card').length === 7 && document.getElementById('status').textContent === ''")
        yield page
        assert not errors, errors
        context.close()
