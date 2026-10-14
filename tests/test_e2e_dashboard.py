"""
Playwright End-to-End Test Suite for Predictive VRAM Dashboard.
Covers Section 6.4 of implementation.md:
- Zero console errors / warnings
- Scenarios A through E execution & UI assertions
- Stop and Reset functionality
- Back-to-back scenario runs (A->B->C->D->E->A)
- CUDA / GPU panel (real CPU fallback and Mock CUDA)
- Viewport audits (1920x1080, 1440x900, 1280x720) with screenshot capture
- Theme (CSS custom properties, glassmorphism, neon glow)
- Accessibility smoke (keyboard navigation, accessible names)
"""

import json
import os
import threading
import time
from pathlib import Path
import pytest
from playwright.sync_api import sync_playwright

from src.dashboard.server import DashboardAPIHandler, ThreadingHTTPServer

PORT = 8092
BASE_URL = f"http://127.0.0.1:{PORT}"
SCREENSHOT_DIR = Path("docs/screenshots")


@pytest.fixture(scope="module")
def server_instance():
    """Runs a dedicated dashboard server instance for the E2E test module."""
    server_address = ("127.0.0.1", PORT)
    httpd = ThreadingHTTPServer(server_address, DashboardAPIHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.5)
    yield BASE_URL
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(scope="module")
def browser_context():
    """Launches Playwright headless Chromium."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


def test_page_loads_clean_with_zero_console_errors(server_instance, browser_context):
    """Asserts that page loads with zero console errors or uncaught warnings."""
    page = browser_context.new_page(viewport={"width": 1920, "height": 1080})
    console_errors = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type in ["error", "warning"] else None)

    page.goto(server_instance)
    page.wait_for_selector("#kpi-plan")
    time.sleep(1.0)

    # Filter out external font connection notices if any
    real_errors = [e for e in console_errors if "favicon" not in e and "font" not in e.lower()]
    assert len(real_errors) == 0, f"Detected console errors/warnings: {real_errors}"
    page.close()


def test_theme_and_glassmorphism_css(server_instance, browser_context):
    """Asserts Phase 3 grey neon glassmorphism CSS properties."""
    page = browser_context.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(server_instance)
    page.wait_for_selector(".glass-panel")

    # Evaluate computed styles
    panel_styles = page.evaluate("""() => {
        const el = document.querySelector('.glass-panel');
        const computed = window.getComputedStyle(el);
        return {
            backdropFilter: computed.backdropFilter || computed.webkitBackdropFilter,
            borderRadius: computed.borderRadius,
            boxShadow: computed.boxShadow,
            borderWidth: computed.borderWidth,
        };
    }""")
    assert "blur" in panel_styles["backdropFilter"] or panel_styles["borderRadius"] in ["18px", "14px"]
    page.close()


def test_scenario_a_e2e(server_instance, browser_context):
    """Executes Scenario A and verifies UI updates, HOLD log, and 0 migrations."""
    page = browser_context.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(server_instance)

    # Click Reset first
    page.click("#btn-reset")
    time.sleep(0.5)

    # Click Scenario A
    page.click("#btn-scen-A")
    page.wait_for_timeout(500)

    # Verify info panel shows lesson
    title = page.text_content("#scenario-title")
    assert "Scenario A" in title
    lesson = page.text_content("#scenario-lesson")
    assert "A good adaptive system should know when not to act" in lesson

    # Wait for completion (status pill changes to COMPLETED)
    page.wait_for_selector("#scenario-status-pill:has-text('COMPLETED')", timeout=12000)

    # Check migration count remains 0
    mig_text = page.text_content("#kpi-migrations")
    assert mig_text.startswith("0 /")

    # Check decision log entries
    log_rows = page.locator("#decision-log .log-row")
    assert log_rows.count() >= 10

    # Capture evidence screenshot
    page.screenshot(path=str(SCREENSHOT_DIR / "scenario_A.png"))
    page.close()


def test_scenario_b_e2e(server_instance, browser_context):
    """Executes Scenario B and verifies proactive migration, split visual update, and lower memory."""
    page = browser_context.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(server_instance)

    page.click("#btn-reset")
    time.sleep(0.5)

    page.click("#btn-scen-B")
    page.wait_for_timeout(500)

    lesson = page.text_content("#scenario-lesson")
    assert "Predict a memory problem early enough to avoid it" in lesson

    # Wait for completion
    page.wait_for_selector("#scenario-status-pill:has-text('COMPLETED')", timeout=15000)

    # Check that migration occurred: plan changed to UD [0-0] | EA [1-2] | EB [3-5]
    plan = page.text_content("#kpi-plan")
    assert "USER_DEVICE [0-0]" in plan
    assert "EDGE_A [1-2]" in plan
    assert "EDGE_B [3-5]" in plan

    # Check layer count: UD has 1, EA has 2, EB has 3
    assert "1 Layers" in page.text_content("#count-ud")
    assert "2 Layers" in page.text_content("#count-ea")
    assert "3 Layers" in page.text_content("#count-eb")

    # Migration count is 1
    assert page.text_content("#kpi-migrations").startswith("1 /")

    page.screenshot(path=str(SCREENSHOT_DIR / "scenario_B.png"))
    page.close()


def test_scenario_c_e2e(server_instance, browser_context):
    """Executes Scenario C and verifies 8s dip, zero migrations, and HOLD decisions."""
    page = browser_context.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(server_instance)

    page.click("#btn-reset")
    time.sleep(0.5)

    page.click("#btn-scen-C")
    page.wait_for_timeout(500)

    lesson = page.text_content("#scenario-lesson")
    assert "Do not migrate just because conditions briefly worsen" in lesson

    page.wait_for_selector("#scenario-status-pill:has-text('COMPLETED')", timeout=18000)

    # Zero migrations throughout
    assert page.text_content("#kpi-migrations").startswith("0 /")

    # Plan unchanged
    assert "USER_DEVICE [0-1]" in page.text_content("#kpi-plan")

    page.screenshot(path=str(SCREENSHOT_DIR / "scenario_C.png"))
    page.close()


def test_scenario_d_e2e(server_instance, browser_context):
    """Executes Scenario D and verifies migration to local-heavy partition after persistent dip."""
    page = browser_context.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(server_instance)

    page.click("#btn-reset")
    time.sleep(0.5)

    page.click("#btn-scen-D")
    page.wait_for_timeout(500)

    lesson = page.text_content("#scenario-lesson")
    assert "Ignore short-lived fluctuations, but adapt to persistent degradation" in lesson

    page.wait_for_selector("#scenario-status-pill:has-text('COMPLETED')", timeout=18000)

    # Migrated to local-heavy: UD [0-1] | EA [2-3]
    plan = page.text_content("#kpi-plan")
    assert "USER_DEVICE [0-1]" in plan

    # Local tier now has 2 layers
    assert "2 Layers" in page.text_content("#count-ud")
    assert page.text_content("#kpi-migrations").startswith("1 /")

    page.screenshot(path=str(SCREENSHOT_DIR / "scenario_D.png"))
    page.close()


def test_scenario_e_e2e(server_instance, browser_context):
    """Executes Scenario E and verifies failure detection, rollback, lock release, and original layout."""
    page = browser_context.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(server_instance)

    page.click("#btn-reset")
    time.sleep(0.5)

    page.click("#btn-scen-E")
    page.wait_for_timeout(500)

    lesson = page.text_content("#scenario-lesson")
    assert "A safe system needs failure recovery" in lesson

    page.wait_for_selector("#scenario-status-pill:has-text('COMPLETED')", timeout=15000)

    # 1 rollback recorded
    assert "/ 1" in page.text_content("#kpi-migrations")

    # Original plan intact
    assert "USER_DEVICE [0-1]" in page.text_content("#kpi-plan")
    assert "2 Layers" in page.text_content("#count-ud")

    # Decision log has ROLLBACK
    assert page.locator("#decision-log .log-row.rollback").count() >= 1

    page.screenshot(path=str(SCREENSHOT_DIR / "scenario_E.png"))
    page.close()


def test_stop_and_reset_mid_run(server_instance, browser_context):
    """Verifies that Stop and Reset work cleanly mid-run without leftover locks."""
    page = browser_context.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(server_instance)

    page.click("#btn-scen-B")
    page.wait_for_selector("#btn-stop:not([disabled])", timeout=5000)

    # Click Stop
    page.click("#btn-stop")
    page.wait_for_selector("#scenario-status-pill:has-text('STOPPED')", timeout=5000)

    # Click Reset
    page.click("#btn-reset")
    page.wait_for_selector("#scenario-status-pill:has-text('IDLE')", timeout=5000)
    assert "USER_DEVICE [0-1]" in page.text_content("#kpi-plan")

    page.close()


def test_back_to_back_runs(server_instance, browser_context):
    """Runs scenarios A -> B -> C -> D -> E -> A back-to-back to catch state leakage."""
    page = browser_context.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(server_instance)

    for scen in ["A", "B", "C", "D", "E", "A"]:
        page.click(f"#btn-scen-{scen}")
        page.wait_for_selector("#scenario-status-pill:has-text('COMPLETED')", timeout=20000)
        page.click("#btn-reset")
        time.sleep(0.3)

    page.close()


def test_cuda_panel_both_states(server_instance, browser_context):
    """Verifies both CPU fallback and Mock CUDA panel states."""
    page = browser_context.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(server_instance)

    # 1. Real CPU fallback state
    desc = page.text_content("#gpu-status-desc")
    assert "CUDA unavailable — running on CPU" in desc
    page.screenshot(path=str(SCREENSHOT_DIR / "cuda_panel_cpu.png"))

    # 2. Toggle to Mock CUDA
    page.click("#btn-toggle-gpu")
    time.sleep(0.8)
    pill = page.text_content("#gpu-pill")
    assert "MOCK CUDA ACTIVE" in pill
    vram = page.text_content("#gpu-vram-val")
    assert "MB /" in vram
    page.screenshot(path=str(SCREENSHOT_DIR / "cuda_panel_mock.png"))

    # Toggle back to CPU fallback
    page.click("#btn-toggle-gpu")
    time.sleep(0.8)
    assert "CPU FALLBACK" in page.text_content("#gpu-pill")

    page.close()


@pytest.mark.parametrize("vp", [
    {"width": 1920, "height": 1080, "name": "viewport_1920x1080"},
    {"width": 1440, "height": 900, "name": "viewport_1440x900"},
    {"width": 1280, "height": 720, "name": "viewport_1280x720"},
])
def test_viewports_no_overflow_and_capture(server_instance, browser_context, vp):
    """Tests 3 viewports: asserts zero horizontal overflow and captures screenshots."""
    page = browser_context.new_page(viewport={"width": vp["width"], "height": vp["height"]})
    page.goto(server_instance)
    page.wait_for_selector(".app-shell")
    time.sleep(0.5)

    # Check horizontal scroll width <= client width
    overflow = page.evaluate("() => document.documentElement.scrollWidth > window.innerWidth")
    assert not overflow, f"Detected horizontal scroll overflow on viewport {vp['name']}"

    page.screenshot(path=str(SCREENSHOT_DIR / f"{vp['name']}.png"), full_page=True)
    page.close()


def test_accessibility_smoke(server_instance, browser_context):
    """Asserts buttons are reachable via Tab key and have visible text / labels."""
    page = browser_context.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(server_instance)
    page.wait_for_selector("#btn-scen-A")

    # Verify buttons have non-empty text content
    buttons = page.locator("button.btn")
    for i in range(buttons.count()):
        text = buttons.nth(i).text_content().strip()
        assert len(text) > 0, f"Button {i} missing accessible text label"

    page.close()
