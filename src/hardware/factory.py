"""
Hardware Provider Factory for Module 11.

Instantiates appropriate HardwareTelemetryProvider based on environment,
runtime configuration, and testing requirements.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from src.hardware.cpu import CPUHardwareProvider
from src.hardware.cuda import CUDAHardwareProvider, MockCUDAHardwareProvider
from src.hardware.provider import HardwareTelemetryProvider


class HardwareProviderFactory:
    """Factory for creating pluggable hardware telemetry providers."""

    @staticmethod
    def create_provider(
        provider_type: str = "auto",
        fallback_to_cpu: bool = True,
        **kwargs: Any,
    ) -> HardwareTelemetryProvider:
        """
        Create a hardware telemetry provider instance.

        Args:
            provider_type: 'auto', 'cpu', 'cuda', or 'mock'.
            fallback_to_cpu: If True, falls back to CPU provider when CUDA is requested but absent.
            **kwargs: Extra arguments passed to provider constructor (e.g. for mock provider).

        Returns:
            Configured HardwareTelemetryProvider instance.
        """
        provider_type = provider_type.lower().strip()

        if provider_type == "auto":
            if torch.cuda.is_available():
                return CUDAHardwareProvider(strict=False)
            return CPUHardwareProvider()

        if provider_type == "cpu":
            return CPUHardwareProvider()

        if provider_type == "cuda":
            if not torch.cuda.is_available() and fallback_to_cpu:
                return CPUHardwareProvider()
            return CUDAHardwareProvider(strict=True)

        if provider_type in ("mock", "mock_cuda"):
            return MockCUDAHardwareProvider(**kwargs)

        raise ValueError(
            f"Unknown hardware provider type '{provider_type}'. "
            f"Supported types: ['auto', 'cpu', 'cuda', 'mock']"
        )

    @staticmethod
    def get_supported_types() -> List[str]:
        return ["auto", "cpu", "cuda", "mock"]
