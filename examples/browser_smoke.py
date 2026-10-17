"""Launch the real app and verify dashboard/backend correspondence in Chromium.

Prerequisites: install .[dev,browser], playwright install chromium, train-all.
No credential is printed or persisted; generated tokens exist only in this process.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/browser-smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    database = args.output / "decisions.sqlite3"
    if database.exists():
        parser.error("choose a fresh --output directory for an independent browser run")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = secrets.token_urlsafe(32)
    env = {
        **os.environ,
        "EDL_OPERATOR_TOKEN": token,
        "EDL_DISPATCHER_TOKEN": secrets.token_urlsafe(32),
        "EDL_CONTROL_MODE": "simulation",
    }
    command = [
        sys.executable,
        "-m",
        "energy_decision_layer",
        "serve",
        "--models",
        str(args.models),
        "--database",
        str(database),
        "--steps",
        "12",
        "--port",
        str(port),
    ]
    url = f"http://127.0.0.1:{port}"
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            for _ in range(200):
                if process.poll() is not None:
                    raise RuntimeError("server exited; inspect server.log")
                try:
                    with urlopen(url + "/ready", timeout=0.5) as response:
                        if response.status == 200:
                            break
                except (OSError, URLError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("server readiness timed out")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors: list[str] = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                expect(page.locator("#connection")).to_contain_text("Connected")
                expect(page.locator("#dispatch")).to_be_disabled()
                expect(page.locator("#overview")).to_contain_text("MW")
                expect(page.locator("#history")).to_contain_text("No dispatch decisions")
                # Fair evaluation and forecast accuracy come from persisted backend reports.
                evaluation = page.request.get(url + "/api/evaluation").json()
                if evaluation.get("metrics.json"):
                    expect(page.locator("#evaluation-comparison")).to_contain_text("learned")
                    expect(page.locator("#forecast-accuracy")).to_contain_text("RMSE")
                    expect(page.locator("#evaluation-scope")).to_contain_text("Synthetic")
                    expect(page.locator("#evaluation-comparison tbody tr")).to_have_count(6)
                    expect(page.locator("#forecast-accuracy tbody tr")).to_have_count(3)
                page.locator("#token").fill(token)
                page.locator("#authenticate").click()
                expect(page.locator("#permissions")).to_contain_text("operator")
                expect(page.locator("#dispatch")).to_be_enabled()
                page.locator("#recommend").click()
                expect(page.locator("#proposal")).to_contain_text("compute_power_mw")
                expect(page.locator("#verdict")).to_contain_text('"executed": false')
                page.locator("#dispatch").click()
                expect(page.locator("#timestamp")).to_contain_text("interval 1")
                expect(page.locator("#history")).to_contain_text("executed")
                # Obtain a CURRENT proposal before testing manual editing and unsafe projection.
                page.locator("#recommend").click()
                expect(page.locator("#proposal")).to_contain_text(":1")
                page.locator("details").first.locator("summary").click()
                proposed = json.loads(page.locator("#manual").input_value())
                proposed.update(
                    compute_power_mw=100.0,
                    cooling_power_mw=100.0,
                    battery_power_mw=100.0,
                    generator_power_mw=100.0,
                    grid_power_mw=100.0,
                )
                page.locator("#manual").fill(json.dumps(proposed))
                page.locator("#validate").click()
                expect(page.locator("#verdict")).to_contain_text('"status": "modified"')
                expect(page.locator("#verdict")).to_contain_text('"executed": false')
                page.locator("#override").click()
                expect(page.locator("#timestamp")).to_contain_text("interval 2")
                expect(page.locator("#verdict")).to_contain_text('"executed": true')
                # Advance far enough to exercise trained forecasting after warm-up.
                for step in (3, 4):
                    page.locator("#dispatch").click()
                    expect(page.locator("#timestamp")).to_contain_text(f"interval {step}")
                expect(page.locator("#forecast-meta")).to_contain_text("gradient-boosted-quantiles")
                state = page.request.get(url + "/api/state").json()
                history = page.request.get(url + "/api/decisions").json()["items"]
                assert state["step_index"] == 4
                assert len(history) == 4
                assert history[0]["status"] == "executed"
                assert page.locator("#overview .card").count() >= 10
                page.screenshot(path=str(args.output / "desktop.png"), full_page=True)
                page.set_viewport_size({"width": 390, "height": 844})
                page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                # Network failure must produce a visible error without inventing fresh state.
                page.route("**/api/**", lambda route: route.abort())
                page.locator("#authenticate").click()
                expect(page.locator("#error")).to_be_visible()
                assert not errors, errors
                browser.close()
                (args.output / "result.json").write_text(
                    json.dumps(
                        {
                            "passed": True,
                            "steps": 4,
                            "browser_errors": errors,
                            "checks": [
                                "read-only defaults",
                                "operator auth",
                                "recommendation preview",
                                "simulated dispatch",
                                "unsafe manual modification",
                                "trained forecasts",
                                "backend/history/UI correspondence",
                                "mobile layout",
                                "network error",
                            ],
                        },
                        indent=2,
                    )
                    + "\n"
                )
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    print(f"Browser smoke passed: {args.output}/result.json")


if __name__ == "__main__":
    main()
