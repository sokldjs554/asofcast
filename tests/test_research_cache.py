"""Exercise the real research loader against a retained pre-deploy response."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_evidence(date):
    app = (ROOT / 'src/asofcast/static/app.js').read_text()
    start = app.index('  async function loadResearchEvidence()')
    end = app.index("  $('reloadResearch').addEventListener", start)
    research = json.loads((ROOT / 'src/asofcast/static/research-evidence.json').read_text())
    research['research_date'] = date
    harness = '''
const nodes = new Map();
const $ = id => {
  if (!nodes.has(id)) nodes.set(id, {textContent:'', children:[],
    replaceChildren(...children){this.children=children;}});
  return nodes.get(id);
};
const document = {createElement: () => ({append(){}})};
const fmt = value => String(value), signed = value => String(value);
let request;
async function requestJSON(url, options) {request={url,options};return payload;}
'''
    code = 'const payload=' + json.dumps(research) + ';\n' + harness + app[start:end]
    code += '''
loadResearchEvidence().then(() => console.log(JSON.stringify({request,
  rows:$('researchRows').children.length, status:$('researchStatus').textContent,
  verdict:$('researchVerdict').textContent, disabled:$('reloadResearch').disabled})));
'''
    result = subprocess.run(['node', '-e', code], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_old_research_cannot_appear_below_current_explanation():
    result = load_evidence('2026-10-02')
    assert result['rows'] == 0
    assert '원문 보고서' in result['status']
    assert result['disabled'] is False


def test_current_research_load_bypasses_retained_response():
    result = load_evidence('2026-10-05')
    assert result['request']['options']['cache'] == 'no-store'
    assert '?v=' in result['request']['url']
    assert result['rows'] == 6
    assert '0 / 6' in result['verdict']
