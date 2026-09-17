"""
RuntimeTrace data container and serialization utilities for Module 9.

Ensures strict reproducibility, temporal causality auditing, and
offline evaluation across Static, Reactive, and Predictive regimes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.orchestration.types import (
    ControlCycle,
    MigrationWindowRecord,
    OrchestrationMetrics,
    PredictionLeadRecord,
    TokenRecord,
)


@dataclass
class RuntimeTrace:
    """
    Comprehensive record of an integrated closed-loop runtime experiment.

    Contains all metadata, individual token latencies, control cycles,
    migration windows, prediction lead records, and final summary metrics.
    """
    metadata: Dict[str, Any] = field(default_factory=dict)
    cycles: List[ControlCycle] = field(default_factory=list)
    token_records: List[TokenRecord] = field(default_factory=list)
    migration_windows: List[MigrationWindowRecord] = field(default_factory=list)
    prediction_leads: List[PredictionLeadRecord] = field(default_factory=list)
    summary: Optional[OrchestrationMetrics] = None

    def add_cycle(self, cycle: ControlCycle) -> None:
        """Append a completed control cycle to the trace."""
        self.cycles.append(cycle)

    def add_token(self, token: TokenRecord) -> None:
        """Append a token generation record."""
        self.token_records.append(token)

    def add_migration_window(self, window: MigrationWindowRecord) -> None:
        """Append a migration context window."""
        self.migration_windows.append(window)

    def add_prediction_lead(self, lead: PredictionLeadRecord) -> None:
        """Append a prediction lead-time record."""
        self.prediction_leads.append(lead)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "cycles": [c.to_dict() for c in self.cycles],
            "token_records": [t.to_dict() for t in self.token_records],
            "migration_windows": [w.to_dict() for w in self.migration_windows],
            "prediction_leads": [l.to_dict() for l in self.prediction_leads],
            "summary": self.summary.to_dict() if self.summary else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuntimeTrace:
        cycles = [ControlCycle.from_dict(c) for c in data.get("cycles", [])]
        tokens = [TokenRecord.from_dict(t) for t in data.get("token_records", [])]
        windows = [MigrationWindowRecord.from_dict(w) for w in data.get("migration_windows", [])]
        leads = [PredictionLeadRecord.from_dict(l) for l in data.get("prediction_leads", [])]
        summary = OrchestrationMetrics.from_dict(data["summary"]) if data.get("summary") else None

        return cls(
            metadata=data.get("metadata", {}),
            cycles=cycles,
            token_records=tokens,
            migration_windows=windows,
            prediction_leads=leads,
            summary=summary,
        )

    def save_json(self, path: Path | str) -> None:
        """Serialize trace to JSON file, creating parent directories if needed."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> RuntimeTrace:
        """Load and deserialize a RuntimeTrace from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
