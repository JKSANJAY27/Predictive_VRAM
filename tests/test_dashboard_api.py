"""
Automated unit tests for Dashboard REST API endpoints and Server-Sent Events (SSE).
Covers Section 6.3 of implementation.md.
"""

import json
import threading
import time
import urllib.request
import pytest

from src.dashboard.scenario_runner import ScenarioExecutionManager
from src.dashboard.server import DashboardAPIHandler, ThreadingHTTPServer


@pytest.fixture(scope="module")
def dashboard_server():
    """Starts a dashboard server instance on an ephemeral port for testing."""
    server_address = ("127.0.0.1", 8089)
    httpd = ThreadingHTTPServer(server_address, DashboardAPIHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    yield "http://127.0.0.1:8089"
    httpd.shutdown()
    httpd.server_close()


def test_api_state_schema(dashboard_server):
    """GET /api/state returns the authoritative schema."""
    req = urllib.request.urlopen(f"{dashboard_server}/api/state")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))

    required_keys = [
        "status",
        "active_scenario",
        "scenario_title",
        "scenario_lesson",
        "current_plan_id",
        "current_plan_str",
        "layers_user_device",
        "layers_edge_a",
        "layers_edge_b",
        "migration_phase",
        "migration_count",
        "rollback_count",
        "bandwidth_mbps",
        "rtt_ms",
        "kv_cache_bytes",
        "memory_usage_mb",
        "decision_log",
        "cuda_available",
        "vram_status",
    ]
    for k in required_keys:
        assert k in data, f"Missing key '{k}' in /api/state payload"


def test_scenario_lifecycle_via_api(dashboard_server):
    """POST /api/scenario/start -> running -> completed -> reset."""
    # 1. Start Scenario A
    start_payload = json.dumps({"scenario": "A", "seed": 42, "step_delay": 0.005}).encode("utf-8")
    req = urllib.request.Request(
        f"{dashboard_server}/api/scenario/start",
        data=start_payload,
        headers={"Content-Type": "application/json"},
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200

    # Wait for completion
    time.sleep(0.3)
    req = urllib.request.urlopen(f"{dashboard_server}/api/state")
    state = json.loads(req.read().decode("utf-8"))
    assert state["status"] == "completed"
    assert state["assertion_passed"] is True
    assert len(state["decision_log"]) == 10

    # 2. Reset
    req = urllib.request.Request(
        f"{dashboard_server}/api/scenario/reset",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200

    req = urllib.request.urlopen(f"{dashboard_server}/api/state")
    state = json.loads(req.read().decode("utf-8"))
    assert state["status"] == "idle"
    assert len(state["decision_log"]) == 0


def test_toggle_mock_cuda_via_api(dashboard_server):
    """POST /api/toggle-mock-cuda toggles mock CUDA state."""
    # Enable mock CUDA
    req = urllib.request.Request(
        f"{dashboard_server}/api/toggle-mock-cuda",
        data=json.dumps({"enabled": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200

    req = urllib.request.urlopen(f"{dashboard_server}/api/state")
    state = json.loads(req.read().decode("utf-8"))
    assert state["mock_cuda_active"] is True
    assert state["vram_status"] == "MOCKED"
    assert "RTX 4090" in state["device_name"]

    # Disable mock CUDA
    req = urllib.request.Request(
        f"{dashboard_server}/api/toggle-mock-cuda",
        data=json.dumps({"enabled": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200

    req = urllib.request.urlopen(f"{dashboard_server}/api/state")
    state = json.loads(req.read().decode("utf-8"))
    assert state["mock_cuda_active"] is False
    assert state["vram_status"] == "UNAVAILABLE"
    assert "CPU" in state["vram_status_text"]


def test_catalog_endpoint(dashboard_server):
    """GET /api/catalog returns structural plan candidates."""
    req = urllib.request.urlopen(f"{dashboard_server}/api/catalog")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert "total_plans" in data
    assert data["total_plans"] > 0
    assert len(data["sample_plans"]) > 0
