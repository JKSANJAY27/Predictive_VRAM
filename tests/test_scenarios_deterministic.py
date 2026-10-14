"""
Automated validation of Scenarios A through E under 10 consecutive seeded runs.
Verifies Section 2 requirements and ensures 100% deterministic reproducibility.
"""

import pytest
from src.dashboard.scenario_runner import ScenarioExecutionManager


@pytest.mark.parametrize("scenario_id", ["A", "B", "C", "D", "E"])
def test_scenario_deterministic_10_runs(scenario_id: str):
    """Run each scenario 10 times consecutively and assert 100% pass rate."""
    pass_count = 0
    total_runs = 10

    for seed in range(total_runs):
        manager = ScenarioExecutionManager()
        # Fast step delay for test speed
        success = manager.start_scenario(scenario_id=scenario_id, seed=100 + seed, step_delay=0.001)
        assert success, f"Failed to start scenario {scenario_id} on run {seed}"

        # Wait for completion
        if manager._worker_thread:
            manager._worker_thread.join(timeout=5.0)

        state = manager.state
        assert state.status == "completed", f"Scenario {scenario_id} status is {state.status}: {state.assertion_message}"
        assert state.assertion_passed is True, f"Scenario {scenario_id} failed on run {seed}: {state.assertion_message}"
        pass_count += 1

    assert pass_count == total_runs, f"Expected {total_runs} passes, got {pass_count}"
