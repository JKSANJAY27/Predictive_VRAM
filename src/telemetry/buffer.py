"""
Thread-safe fixed-capacity ring buffer for TelemetrySnapshot records.

The buffer stores up to `capacity` snapshots in insertion order.
When full, the oldest snapshot is evicted to make room for the newest.

Thread safety is provided by a single threading.Lock, which is compatible
with the synchronous generation loop used in this project. No asyncio,
no queues, no multiprocessing shared memory.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Dict, List, Optional

from src.telemetry.types import TelemetrySnapshot


class TelemetryBuffer:
    """
    A thread-safe, fixed-capacity circular buffer for TelemetrySnapshot objects.

    Snapshots are stored in chronological order (oldest at index 0, newest at -1).
    When capacity is exceeded, the oldest snapshot is automatically evicted.

    Attributes:
        capacity: Maximum number of snapshots retained (default 256).

    Usage:
        buffer = TelemetryBuffer(capacity=128)
        buffer.push(snapshot)
        recent = buffer.get_recent(10)   # Last 10 snapshots
        all_snaps = buffer.get_all()
        buffer.clear()
    """

    def __init__(self, capacity: int = 256) -> None:
        if capacity < 1:
            raise ValueError(f"TelemetryBuffer capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._buffer: deque[TelemetrySnapshot] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of snapshots stored before eviction."""
        return self._capacity

    def push(self, snapshot: TelemetrySnapshot) -> None:
        """
        Append a snapshot to the buffer.

        If the buffer is at capacity, the oldest snapshot is silently evicted.

        Args:
            snapshot: A TelemetrySnapshot instance to store.

        Raises:
            TypeError: If snapshot is not a TelemetrySnapshot.
        """
        if not isinstance(snapshot, TelemetrySnapshot):
            raise TypeError(
                f"TelemetryBuffer.push() expects a TelemetrySnapshot, "
                f"got {type(snapshot).__name__}"
            )
        with self._lock:
            self._buffer.append(snapshot)

    def get_recent(self, n: int) -> List[TelemetrySnapshot]:
        """
        Return the last n snapshots in chronological order (oldest first).

        If fewer than n snapshots exist, returns all available snapshots.

        Args:
            n: Number of most-recent snapshots to retrieve.

        Returns:
            List of TelemetrySnapshot, oldest first.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        with self._lock:
            buf = list(self._buffer)
        return buf[-n:] if n > 0 else []

    def get_all(self) -> List[TelemetrySnapshot]:
        """
        Return all stored snapshots in chronological order (oldest first).

        Returns:
            List of all TelemetrySnapshot objects in the buffer.
        """
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Remove all snapshots from the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        """Current number of stored snapshots."""
        with self._lock:
            return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"TelemetryBuffer(capacity={self._capacity}, "
            f"stored={len(self)})"
        )

    # ------------------------------------------------------------------
    # Optional DataFrame export (requires pandas)
    # ------------------------------------------------------------------

    def as_dataframe(self) -> Any:
        """
        Export buffer contents as a flat pandas DataFrame.

        Each row represents one TelemetrySnapshot. Nested fields are
        flattened to columns using dot-notation (e.g., 'memory.ram_used_mb').

        Returns:
            pandas.DataFrame with one row per snapshot.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for TelemetryBuffer.as_dataframe(). "
                "Install it with: pip install pandas"
            ) from exc

        rows = [_flatten_snapshot(s) for s in self.get_all()]
        return pd.DataFrame(rows)

    def summary_dict(self) -> Dict[str, Any]:
        """
        Return a lightweight summary of buffer state without requiring pandas.

        Returns:
            Dict with count, capacity, step range, and timestamp range.
        """
        with self._lock:
            buf = list(self._buffer)

        if not buf:
            return {
                "count": 0,
                "capacity": self._capacity,
                "step_range": None,
                "timestamp_range": None,
            }

        return {
            "count": len(buf),
            "capacity": self._capacity,
            "step_range": (buf[0].step, buf[-1].step),
            "timestamp_range": (buf[0].timestamp, buf[-1].timestamp),
        }


# ---------------------------------------------------------------------------
# Internal helper — flatten a TelemetrySnapshot to a 1-level dict
# ---------------------------------------------------------------------------

def _flatten_snapshot(snapshot: TelemetrySnapshot) -> Dict[str, Any]:
    """
    Flatten a TelemetrySnapshot to a single-level dict for DataFrame export.

    Only scalar values and tagged-value .value fields are included;
    complex nested structures are represented as string summaries.
    """
    row: Dict[str, Any] = {
        "step": snapshot.step,
        "timestamp": snapshot.timestamp,
    }

    # Memory
    if snapshot.memory is not None:
        m = snapshot.memory
        row["mem.ram_used_mb"] = m.ram_used_mb.value
        row["mem.ram_available_mb"] = m.ram_available_mb.value
        row["mem.ram_percent"] = m.ram_percent.value
        row["mem.vram_used_mb"] = m.vram_used_mb.value  # None on CPU machine
        row["mem.process_ram_mb"] = m.process_ram_mb.value

    # CPU
    if snapshot.cpu is not None:
        c = snapshot.cpu
        row["cpu.percent"] = c.cpu_percent_overall.value
        row["cpu.count_logical"] = c.cpu_count_logical.value
        row["cpu.freq_mhz"] = c.cpu_freq_mhz.value

    # KV Cache
    if snapshot.kv_cache is not None:
        kv = snapshot.kv_cache
        row["kv.total_kv_bytes"] = kv.total_kv_bytes.value
        row["kv.total_kv_mb"] = kv.total_kv_mb.value
        row["kv.num_cached_layers"] = kv.num_cached_layers.value
        row["kv.growth_bytes"] = kv.kv_growth_bytes_since_last.value

    # Network
    if snapshot.network is not None:
        n = snapshot.network
        row["net.rtt_ms"] = n.rtt_ms.value
        row["net.bandwidth_mbps"] = n.bandwidth_mbps.value
        row["net.packet_loss"] = n.packet_loss_rate.value
        row["net.scenario"] = n.scenario_label

    # Activations — summarised (multiple per step possible)
    row["activations.count"] = len(snapshot.activations)
    if snapshot.activations:
        total_act_bytes = sum(
            a.byte_size.value or 0 for a in snapshot.activations
        )
        row["activations.total_bytes"] = total_act_bytes

    return row
