"""Repeat the current M2 browser suite and retain reports, including failures.

The browser renders real assets and calls real FastAPI/PyTorch through a local
bridge. This is deliberately NOT reported as public HTTP or Render verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, help='Verified model bundle; omitted: small synthetic test fixture')
    parser.add_argument('--browser', help='Optional Chromium executable; otherwise Playwright installation')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--out', type=Path, default=Path('artifacts/browser-audit'))
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error('--repeats must be between 1 and 10')
    if args.artifacts and not args.artifacts.is_dir():
        parser.error('--artifacts must name an existing model bundle directory')
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    if args.browser:
        env['ASOFCAST_BROWSER'] = args.browser
    if args.artifacts:
        env['ASOFCAST_TEST_ARTIFACTS'] = str(args.artifacts.resolve())
    else:
        env.pop('ASOFCAST_TEST_ARTIFACTS', None)
    results = []
    for repeat in range(1, args.repeats + 1):
        folder = out / f'repeat-{repeat}'
        folder.mkdir(exist_ok=True)
        xml = folder / 'junit.xml'
        env['ASOFCAST_CAPTURE_DIR'] = str(folder)
        command = [sys.executable, '-m', 'pytest', str(root / 'browser_tests'),
                   '-q', f'--junitxml={xml}']
        started = time.monotonic()
        with (folder / 'pytest.log').open('w', encoding='utf-8') as log:
            try:
                completed = subprocess.run(command, cwd=root, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=300)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                log.write('\nBrowser suite exceeded the 300 second timeout.\n')
                code = 124
        counts = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
        if xml.exists():
            for suite in ET.parse(xml).getroot().iter('testsuite'):
                for key in counts:
                    counts[key] += int(suite.get(key, 0))
        success = code == 0 and counts['tests'] > 0 and not any(
            counts[key] for key in ('failures', 'errors', 'skipped'))
        results.append({'repeat': repeat, 'return_code': code, **counts,
                        'seconds': round(time.monotonic() - started, 3), 'passed': success})
        print(json.dumps(results[-1]), flush=True)
        if not success:
            break
    passed = len(results) == args.repeats and all(item['passed'] for item in results)
    files = ['src/asofcast/static/app.js', 'src/asofcast/static/index.html',
             'src/asofcast/static/style.css']
    report = {
        'status': 'passed' if passed else 'failed', 'requested_repeats': args.repeats,
        'transport': 'local Chromium DOM + real FastAPI/PyTorch TestClient bridge',
        'public_http_verified': False, 'python': sys.version.split()[0],
        'model_bundle': str(args.artifacts.resolve()) if args.artifacts else 'generated synthetic test fixture',
        'static_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
        'runs': results,
    }
    (out / 'browser-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
