from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def wait_for_m2(page, url: str, attempts: int = 24) -> None:
    last = None
    for _ in range(attempts):
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            last = None if response is None else response.status
            if response is not None and response.status < 400:
                page.wait_for_timeout(1500)
                if page.locator("h1").count() and "AI Decision Console" in page.locator("h1").inner_text():
                    page.wait_for_function(
                        "() => document.querySelectorAll('#sensorMapGrid .sensor-card').length >= 7"
                        " && document.getElementById('status').textContent === ''",
                        timeout=60_000,
                    )
                    return
        except Exception as exc:  # deployment transition can briefly reset the connection
            last = str(exc)
        page.wait_for_timeout(15_000)
    raise RuntimeError(f"M2 public demo did not become ready: {last}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the public AsOfCast M2 acquisition demo")
    parser.add_argument("--url", default="https://asofcast.onrender.com")
    parser.add_argument("--out", type=Path, default=Path("docs/assets/live-m2"))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
            device_scale_factor=1,
            record_video_dir=str(out),
            record_video_size={"width": 1280, "height": 720},
        )
        page = context.new_page()
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        wait_for_m2(page, args.url)
        page.screenshot(path=str(out / "m2-decision-console.png"), full_page=False)
        page.wait_for_timeout(1800)

        recommended = page.locator("#acquireRecommended")
        acquisition_click_mode = None
        if recommended.is_enabled():
            sensor = recommended.get_attribute("data-sensor")
            acquisition_click_mode = "recommended"
            recommended.click()
        else:
            # The default replay case may legitimately recommend WAIT. For the
            # portfolio story, exercise the user-driven counterfactual path by
            # actively pulling the highest-ranked eligible candidate instead.
            candidate_card = page.locator("#sensorMapGrid .sensor-card.is-candidate").first
            candidate_button = candidate_card.locator(".sensor-acquire")
            if candidate_button.count() and candidate_button.is_enabled():
                sensor = candidate_card.locator(".sensor-card-head strong").inner_text()
                acquisition_click_mode = "manual_top_candidate"
                candidate_button.click()
            else:
                sensor = None
        if sensor:
            page.wait_for_function(
                "() => document.getElementById('status').textContent === ''",
                timeout=30_000,
            )
            page.wait_for_timeout(2600)
            page.screenshot(path=str(out / "m2-after-acquisition.png"), full_page=False)

        page.locator("#counterfactual").scroll_into_view_if_needed()
        page.wait_for_timeout(1600)
        page.screenshot(path=str(out / "m2-counterfactual.png"), full_page=False)

        page.locator("#revision").scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out / "m2-revision.png"), full_page=False)

        page.locator("#pareto").scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out / "m2-pareto.png"), full_page=False)

        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out / "m2-full.png"), full_page=True)

        video = page.video
        context.close()
        browser.close()
        source_video = Path(video.path())
        target_video = out / "asofcast-m2-demo.webm"
        if source_video.resolve() != target_video.resolve():
            shutil.move(str(source_video), target_video)

    if page_errors:
        raise RuntimeError(f"browser page errors: {page_errors}")
    required = [
        "m2-decision-console.png", "m2-after-acquisition.png" if sensor else None,
        "m2-counterfactual.png", "m2-revision.png", "m2-pareto.png",
        "m2-full.png", "asofcast-m2-demo.webm",
    ]
    missing = [name for name in required if name and not (out / name).exists()]
    if missing:
        raise RuntimeError(f"missing capture outputs: {missing}")
    report = {
        "status": "captured",
        "source_url": args.url,
        "captured_at_unix": int(time.time()),
        "browser": "Chromium via Playwright",
        "sensor_clicked": sensor,
        "acquisition_click_mode": acquisition_click_mode,
        "screenshots": [name for name in required if name and name.endswith(".png")],
        "video": "asofcast-m2-demo.webm",
        "scopes": [
            "public Render service, not a mock page",
            "active acquisition changes the stateless replay state",
            "retrospective counterfactual truth remains audit-only",
        ],
    }
    (out / "capture-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
