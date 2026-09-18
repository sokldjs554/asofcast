from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def wait_ready(page) -> None:
    page.wait_for_function(
        """() => {
          const status = document.getElementById('status');
          const forecast = document.getElementById('forecast');
          return status && forecast && status.textContent === '' && forecast.textContent !== '—';
        }""",
        timeout=120_000,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture real AsOfCast live demo media")
    parser.add_argument("--url", default="https://asofcast.onrender.com")
    parser.add_argument("--out", type=Path, default=Path("docs/assets/live"))
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
            device_scale_factor=1,
            record_video_dir=str(out),
            record_video_size={"width": 1280, "height": 720},
        )
        page = context.new_page()
        page.on("pageerror", lambda exc: errors.append(str(exc)))

        response = page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
        if response is None or response.status >= 400:
            raise RuntimeError(f"demo navigation failed: {None if response is None else response.status}")
        wait_ready(page)

        page.screenshot(path=str(out / "portfolio-hero.png"), full_page=False)
        page.screenshot(path=str(out / "demo-full.png"), full_page=True)
        page.wait_for_timeout(1800)

        page.locator("#nextCase").click()
        page.wait_for_function(
            "() => document.getElementById('caseId').value === '1' && document.getElementById('status').textContent === ''",
            timeout=30_000,
        )
        page.wait_for_timeout(1800)

        page.locator("#scenario").select_option("outage")
        page.wait_for_function(
            "() => document.getElementById('status').textContent === '' && !document.getElementById('runButton').disabled",
            timeout=30_000,
        )
        page.wait_for_timeout(1800)
        page.screenshot(path=str(out / "demo-outage.png"), full_page=False)

        page.locator("#stepRange").fill("2")
        page.locator("#stepRange").dispatch_event("input")
        page.wait_for_timeout(1800)

        page.locator("#evidence").scroll_into_view_if_needed()
        page.wait_for_timeout(2200)
        page.screenshot(path=str(out / "demo-evidence.png"), full_page=False)

        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        page.wait_for_timeout(1800)

        video = page.video
        context.close()
        browser.close()

        original_video = Path(video.path())
        target_video = out / "asofcast-demo.webm"
        if original_video.resolve() != target_video.resolve():
            shutil.move(str(original_video), target_video)

    if errors:
        raise RuntimeError(f"browser page errors: {errors}")

    required = [
        "portfolio-hero.png",
        "demo-full.png",
        "demo-outage.png",
        "demo-evidence.png",
        "asofcast-demo.webm",
    ]
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        raise RuntimeError(f"missing capture outputs: {missing}")

    report = {
        "status": "captured",
        "source_url": args.url,
        "captured_at_unix": int(time.time()),
        "browser": "Chromium via Playwright",
        "screenshots": required[:-1],
        "video": required[-1],
        "notes": [
            "navigation used the public Render service",
            "screenshots and video show model-backed live UI interactions",
            "cold-start time is excluded from the edited portfolio narrative but not fabricated",
        ],
    }
    (out / "capture-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
