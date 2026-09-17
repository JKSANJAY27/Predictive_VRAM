"""
Model metadata and tier capability definitions for partition feasibility evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.runtime.tier import TierId
from src.telemetry.types import DataSource


@dataclass(frozen=True)
class ModelMetadata:
    """
    Structural and parameter dimensions of the target transformer model.

    Enables memory requirement projection without repeatedly inspecting weights.
    """
    total_layers: int
    hidden_size: int
    num_heads: int
    vocab_size: int = 50257
    dtype: str = "float32"
    dtype_bytes: int = 4
    total_parameters: int = 124439808
    per_layer_params: Optional[List[int]] = None
    embeddings_params: Optional[int] = None
    head_params: Optional[int] = None

    def __post_init__(self) -> None:
        if self.total_layers <= 0:
            raise ValueError(f"total_layers must be positive, got {self.total_layers}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def synthetic_default(cls, total_layers: int = 12) -> ModelMetadata:
        """Standard 12-layer GPT-2 (124M) default metadata."""
        # GPT-2 base: ~7.08M params per transformer block
        layer_params = 7087872
        emb_params = 38597376  # wte (50257 * 768) + wpe (1024 * 768)
        head_params = 38597376 # lm_head tied or separate
        per_layer = [layer_params] * total_layers
        total_p = emb_params + layer_params * total_layers + 768 * 2
        return cls(
            total_layers=total_layers,
            hidden_size=768,
            num_heads=12,
            vocab_size=50257,
            dtype="float32",
            dtype_bytes=4,
            total_parameters=total_p,
            per_layer_params=per_layer,
            embeddings_params=emb_params,
            head_params=head_params,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelMetadata:
        return cls(
            total_layers=data["total_layers"],
            hidden_size=data["hidden_size"],
            num_heads=data["num_heads"],
            vocab_size=data.get("vocab_size", 50257),
            dtype=data.get("dtype", "float32"),
            dtype_bytes=data.get("dtype_bytes", 4),
            total_parameters=data.get("total_parameters", 124439808),
            per_layer_params=data.get("per_layer_params"),
            embeddings_params=data.get("embeddings_params"),
            head_params=data.get("head_params"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_layers": self.total_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "dtype": self.dtype,
            "dtype_bytes": self.dtype_bytes,
            "total_parameters": self.total_parameters,
            "per_layer_params": self.per_layer_params,
            "embeddings_params": self.embeddings_params,
            "head_params": self.head_params,
        }


@dataclass(frozen=True)
class TierCapacity:
    """
    Capacity attributes and configuration limits of an execution tier.

    Separates static/configured host limits from dynamic RuntimeState observations.
    """
    tier_id: TierId
    device: str = "cpu"
    total_memory_mb: Optional[float] = None
    available_memory_mb: Optional[float] = None
    memory_provenance: DataSource = DataSource.UNAVAILABLE
    compute_capacity_score: float = 1.0
    max_layers: Optional[int] = None
    is_enabled: bool = True

    def __post_init__(self) -> None:
        if self.total_memory_mb is not None and self.total_memory_mb < 0:
            raise ValueError(f"total_memory_mb cannot be negative: {self.total_memory_mb}")
        if self.available_memory_mb is not None and self.available_memory_mb < 0:
            raise ValueError(f"available_memory_mb cannot be negative: {self.available_memory_mb}")
        if self.compute_capacity_score <= 0:
            raise ValueError(f"compute_capacity_score must be > 0: {self.compute_capacity_score}")
        if self.max_layers is not None and self.max_layers < 0:
            raise ValueError(f"max_layers cannot be negative: {self.max_layers}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_id": self.tier_id.value,
            "device": self.device,
            "total_memory_mb": self.total_memory_mb,
            "available_memory_mb": self.available_memory_mb,
            "memory_provenance": self.memory_provenance.value,
            "compute_capacity_score": self.compute_capacity_score,
            "max_layers": self.max_layers,
            "is_enabled": self.is_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TierCapacity:
        return cls(
            tier_id=TierId(data["tier_id"]),
            device=data.get("device", "cpu"),
            total_memory_mb=data.get("total_memory_mb"),
            available_memory_mb=data.get("available_memory_mb"),
            memory_provenance=DataSource(data.get("memory_provenance", DataSource.UNAVAILABLE.value)),
            compute_capacity_score=data.get("compute_capacity_score", 1.0),
            max_layers=data.get("max_layers"),
            is_enabled=data.get("is_enabled", True),
        )
