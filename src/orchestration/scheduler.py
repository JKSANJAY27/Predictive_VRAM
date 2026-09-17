"""
Control loop scheduler enforcing token-boundary and time-boundary synchronization.
"""

from __future__ import annotations

import threading
from typing import Optional

from src.orchestration.config import OrchestrationConfig


class ControlScheduler:
    """
    Determines when control cycles should be triggered relative to token generation boundaries.

    Guarantees:
    - Token boundary evaluation (never interrupts mid-forward-pass).
    - Configurable token interval (e.g., every K generated tokens).
    - Optional wall-clock interval guard (e.g., at least T seconds elapsed).
    - Single-flight locking to prevent overlapping control/migration cycles.
    """

    def __init__(self, config: OrchestrationConfig) -> None:
        self.config = config
        self._last_cycle_step: int = -1
        self._last_cycle_time: float = 0.0
        self._active_lock = threading.Lock()
        self._is_cycle_in_flight: bool = False

    def should_trigger(self, step: int, timestamp: float) -> bool:
        """
        Check if current step or timestamp satisfies the control opportunity criteria.
        """
        # If a cycle is currently running, do not trigger concurrently
        if self._is_cycle_in_flight:
            return False

        # First token check: trigger at step 0 if configured, or wait until interval
        if self._last_cycle_step == -1:
            # Trigger initial baseline cycle at step 0
            return True

        # Token-based trigger check
        tokens_since_last = step - self._last_cycle_step
        if tokens_since_last >= self.config.control_interval_tokens:
            # If time interval is also specified, both must be satisfied (or time alone)
            if self.config.control_interval_seconds is not None:
                time_since_last = timestamp - self._last_cycle_time
                if time_since_last < self.config.control_interval_seconds:
                    return False
            return True

        # Time-based standalone trigger check
        if self.config.control_interval_seconds is not None:
            time_since_last = timestamp - self._last_cycle_time
            if time_since_last >= self.config.control_interval_seconds:
                return True

        return False

    def mark_cycle_started(self) -> bool:
        """Acquire lock and flag cycle as in flight."""
        acquired = self._active_lock.acquire(blocking=False)
        if acquired:
            self._is_cycle_in_flight = True
        return acquired

    def mark_cycle_completed(self, step: int, timestamp: float) -> None:
        """Record completed cycle time and step, and release lock."""
        self._last_cycle_step = step
        self._last_cycle_time = timestamp
        self._is_cycle_in_flight = False
        try:
            self._active_lock.release()
        except RuntimeError:
            pass

    def reset(self) -> None:
        """Reset scheduler history."""
        self._last_cycle_step = -1
        self._last_cycle_time = 0.0
        self._is_cycle_in_flight = False
