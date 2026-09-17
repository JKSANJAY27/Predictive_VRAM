"""
Thread-safe rolling StateBuffer for RuntimeState sequences.

Maintains a fixed-capacity ring buffer of timestamp-ordered RuntimeState instances.
Supports windowed queries, serialization to/from JSON for trace replay,
and feature matrix extraction.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Union

from src.state.features import FeatureExtractor, FeatureVector
from src.state.types import RuntimeState


class StateBuffer:
    """
    Rolling ring buffer for RuntimeState objects.

    Guarantees:
    - Fixed memory footprint bounded by `capacity`.
    - Strict chronological order enforcement (rejects out-of-order arrivals).
    - Thread-safe append and access operations via reentrant/standard Lock.
    - JSON serialization and deserialization for offline analysis and replay.
    """

    def __init__(self, capacity: int = 128) -> None:
        if capacity <= 0:
            raise ValueError(f"StateBuffer capacity must be > 0, got {capacity}")
        self._capacity = capacity
        self._buffer: deque[RuntimeState] = deque(maxlen=capacity)
        self._lock = Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def append(self, state: RuntimeState) -> None:
        """
        Append a new RuntimeState.

        Raises:
            ValueError: If state.timestamp is strictly less than the latest
                        timestamp in the buffer (out-of-order rejection).
        """
        with self._lock:
            if self._buffer:
                latest_t = self._buffer[-1].timestamp
                if state.timestamp < latest_t:
                    raise ValueError(
                        f"Out-of-order state rejected: timestamp {state.timestamp} "
                        f"is earlier than buffer latest {latest_t}"
                    )
                if state.timestamp == latest_t and state.step_index < self._buffer[-1].step_index:
                    raise ValueError(
                        f"Out-of-order state rejected: step_index {state.step_index} "
                        f"is earlier than buffer latest {self._buffer[-1].step_index}"
                    )
            self._buffer.append(state)

    def latest(self) -> Optional[RuntimeState]:
        """Return the most recently appended RuntimeState, or None if empty."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_recent(self, n: int) -> List[RuntimeState]:
        """
        Return the `n` most recent RuntimeStates in chronological order.
        If `n` exceeds current size, returns all available states without padding or error.
        """
        if n <= 0:
            return []
        with self._lock:
            count = min(n, len(self._buffer))
            # Slice from rightmost elements
            items = list(self._buffer)
            return items[-count:]

    def get_all(self) -> List[RuntimeState]:
        """Return all stored RuntimeStates in chronological order."""
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Empty the buffer."""
        with self._lock:
            self._buffer.clear()

    def get_window(self, size: int) -> Dict[str, Any]:
        """
        Retrieve a state window up to `size` with completeness metadata.

        Does NOT pad or raise if fewer states are available than `size`.
        """
        states = self.get_recent(size)
        return {
            "states": states,
            "requested_length": size,
            "actual_length": len(states),
            "is_partial": len(states) < size,
        }

    def to_feature_vectors(self, n: Optional[int] = None) -> List[FeatureVector]:
        """Extract FeatureVectors from recent states (all if n is None)."""
        states = self.get_all() if n is None else self.get_recent(n)
        return FeatureExtractor.from_states(states)

    def to_feature_matrix(
        self,
        n: Optional[int] = None,
        fill_value: Optional[float] = None,
        as_numpy: bool = False,
    ) -> Any:
        """Extract a 2D feature matrix from buffer states."""
        states = self.get_all() if n is None else self.get_recent(n)
        return FeatureExtractor.to_matrix(states, fill_value=fill_value, as_numpy=as_numpy)

    def to_dataframe(self) -> Any:
        """
        Export buffer states to a pandas DataFrame if pandas is installed.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError("pandas is required for to_dataframe()") from err

        states = self.get_all()
        if not states:
            return pd.DataFrame(columns=list(FeatureExtractor.FEATURE_NAMES))

        vectors = FeatureExtractor.from_states(states)
        records = [vec.to_dense(fill_value=float("nan")) for vec in vectors]
        return pd.DataFrame(records, columns=list(FeatureExtractor.FEATURE_NAMES))

    def summary_dict(self) -> Dict[str, Any]:
        """Return a lightweight status summary of the buffer."""
        with self._lock:
            count = len(self._buffer)
            if count == 0:
                return {
                    "capacity": self._capacity,
                    "count": 0,
                    "time_span_s": 0.0,
                    "earliest_step": None,
                    "latest_step": None,
                }
            earliest = self._buffer[0]
            latest = self._buffer[-1]
            return {
                "capacity": self._capacity,
                "count": count,
                "time_span_s": float(latest.timestamp - earliest.timestamp),
                "earliest_step": earliest.step_index,
                "latest_step": latest.step_index,
            }

    # -----------------------------------------------------------------------
    # Serialization & Trace Replay
    # -----------------------------------------------------------------------

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Serialize all states in the buffer to a JSON file."""
        states = self.get_all()
        data = {
            "capacity": self._capacity,
            "count": len(states),
            "states": [s.to_dict() for s in states],
        }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_json(self, filepath: Union[str, Path]) -> None:
        """
        Load states from a JSON file into the buffer, replacing existing contents.
        Validates chronological order.
        """
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_states = data.get("states", [])
        parsed_states = [RuntimeState.from_dict(item) for item in raw_states]

        with self._lock:
            self._buffer.clear()
            for s in parsed_states:
                self.append(s)

    @classmethod
    def from_json_file(cls, filepath: Union[str, Path], capacity: Optional[int] = None) -> StateBuffer:
        """Construct a new StateBuffer and populate it from a JSON file."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cap = capacity if capacity is not None else data.get("capacity", 128)
        buf = cls(capacity=cap)
        for item in data.get("states", []):
            buf.append(RuntimeState.from_dict(item))
        return buf
