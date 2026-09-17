import argparse
import json
from fastapi.testclient import TestClient
from asofcast.service import create_app
from pathlib import Path
from playwright.sync_api import sync_playwright
parser = argparse.ArgumentParser(description='Offline Chromium DOM + real ASGI verification; no network navigation')
parser.add_argument('--browser', help='Optional Chromium executable; otherwise use the Playwright installation')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=args.browser, headless=True, args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1120},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda e: errors.append(str(e)))
    failed=[]
    page.on('response',lambda r:failed.append({'status':r.status,'url':r.url}) if r.status>=400 else None)
    client = TestClient(create_app(root / 'artifacts/demo'))
    def asgi_fetch(source, url):
        response = client.get(url)
        return {'status': response.status_code, 'body': response.text}
    page.expose_binding('asgiFetch', asgi_fetch)
    page.set_content((root / 'src/asofcast/static/index.html').read_text().replace('<link rel="stylesheet" href="/static/style.css">','').replace('<script src="/static/app.js" defer></script>',''))
    page.add_style_tag(content=(root / 'src/asofcast/static/style.css').read_text())
    page.evaluate("""window.fetch = async (url, options = {}) => {
        if (options.signal && options.signal.aborted) throw new DOMException('Aborted', 'AbortError');
        const r = await window.asgiFetch(String(url));
        return {ok: r.status >= 200 && r.status < 300, status:r.status,
                json:async()=>JSON.parse(r.body), text:async()=>r.body};
    };""")
    page.add_script_tag(content=(root / 'src/asofcast/static/app.js').read_text())
    page.wait_for_function("document.getElementById('sensors').children.length === 7")
    assert '합성' in page.locator('#sourceBadge').inner_text()
    assert page.locator('#results tr').count()==11
    assert not page.locator('#status').inner_text()
    page.screenshot(path=str(root/'docs/assets/dashboard-desktop.png'),full_page=True)
    page.locator('#nextCase').click()
    page.wait_for_function("document.getElementById('caseId').value === '1' && document.getElementById('status').textContent === ''")
    assert page.locator('#caseId').input_value()=='1'
    page.locator('#scenario').select_option('outage')
    page.wait_for_function("document.getElementById('status').textContent === '' && !document.getElementById('runButton').disabled")
    page.locator('#stepRange').fill('2')
    page.locator('#stepRange').dispatch_event('input')
    assert page.locator('#stepLabel').inner_text()=='60분'
    page.locator('#caseId').fill('99999')
    page.locator('#runButton').click()
    assert '정수로 입력' in page.locator('#status').inner_text()
    page.locator('#caseId').fill('0')
    page.locator('#scenario').select_option('mixed')
    page.wait_for_function("document.getElementById('status').textContent === '' && !document.getElementById('runButton').disabled")
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(root/'docs/assets/dashboard-mobile.png'),full_page=True)
    overflow=page.evaluate('document.documentElement.scrollWidth > window.innerWidth')
    assert not overflow, 'mobile viewport overflow'
    assert not errors,errors
    assert not failed,failed
    result={'browser':'Chromium', 'transport':'offline DOM assets + real FastAPI TestClient via JS/Python bridge', 'network_navigation_verified':False, 'limitation':'Direct Chromium localhost navigation blocked by administrator; no settings changed','desktop':[1440,1120],'mobile':[390,844],
      'checks':['model-loaded','synthetic-label','7 sensors','11 metric rows','next case','outage scenario','step slider','invalid-case validation','mobile no overflow'],
      'page_errors':errors,'failed_browser_resource_requests':failed,'status':'passed'}
    (root/'docs/reports/browser_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    browser.close()
