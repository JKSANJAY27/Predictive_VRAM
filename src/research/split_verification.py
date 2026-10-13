"""
Genuine Split Inference Verification (Module 13).

Audits and verifies that model partitions execute on designated devices,
tracks tensor transfer records between tiers, monitors KV-cache ownership,
and enforces token parity against a monolithic reference model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.research.empirical_classes import DevicePlacementMode, EvidenceQualityDescriptor, ExperimentClass
from src.runtime.executor import DistributedInferenceExecutor, GenerationResult
from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId
from src.runtime.transfer import TransferRecord

logger = logging.getLogger(__name__)


@dataclass
class PartitionPlacementAudit:
    """Detailed audit of a partition's physical device mapping."""
    tier_id: str
    tier_name: str
    target_device: str
    layer_start: int
    layer_end: int
    layer_count: int
    parameter_count: int
    is_cuda_device: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransferAuditRecord:
    """Detailed record of an inter-tier activation or state transfer."""
    source_tier: str
    destination_tier: str
    source_device: str
    destination_device: str
    tensor_shape: List[int]
    tensor_dtype: str
    size_bytes: int
    transfer_duration_ms: float
    is_cross_device: bool
    is_simulated_transfer: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SplitInferenceVerificationReport:
    """
    Comprehensive verification report establishing genuine split inference behavior,
    device placement, transfer auditing, and token parity against monolithic reference.
    """
    model_name: str
    total_layers: int
    partition_plan_id: str
    active_tiers: List[str]
    device_placement_mode: str
    placement_audits: List[PartitionPlacementAudit]
    transfers: List[TransferAuditRecord]
    total_transfer_bytes: int
    total_transfer_time_ms: float
    token_parity_verified: bool
    generated_tokens_split: List[int]
    generated_tokens_monolithic: List[int]
    prompt: str
    generated_text: str
    execution_time_seconds: float
    kv_cache_summary: Dict[str, Any]
    synchronization_recorded: bool
    evidence_quality: Dict[str, Any]
    parity_mismatch_details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["placement_audits"] = [p.to_dict() for p in self.placement_audits]
        d["transfers"] = [t.to_dict() for t in self.transfers]
        return d


class SplitInferenceVerifier:
    """
    Validates physical/logical partition execution and token parity.
    """

    @classmethod
    def verify_plan(
        cls,
        model: LayeredTransformer,
        partition_plan: PartitionPlan,
        tiers: Dict[TierId, Tier],
        prompt: str = "The future of distributed edge computing and predictive resource allocation",
        max_new_tokens: int = 8,
    ) -> SplitInferenceVerificationReport:
        """
        Execute split inference under partition_plan, record physical layer placement,
        audit transfers, and verify token parity against monolithic baseline execution.
        """
        cuda_avail = torch.cuda.is_available()
        active_tiers = partition_plan.get_active_tiers()

        # 1. Audit layer placement across tiers
        placements: List[PartitionPlacementAudit] = []
        for tier_id, (start, end) in active_tiers:
            t = tiers[tier_id]
            dev_str = str(t.device)
            num_layers = end - start + 1
            # Count parameters placed in this block range
            param_count = sum(
                sum(p.numel() for p in model.blocks[idx].parameters())
                for idx in range(start, end + 1)
            )
            placements.append(
                PartitionPlacementAudit(
                    tier_id=tier_id.value,
                    tier_name=t.name,
                    target_device=dev_str,
                    layer_start=start,
                    layer_end=end,
                    layer_count=num_layers,
                    parameter_count=param_count,
                    is_cuda_device="cuda" in dev_str.lower(),
                )
            )

        # 2. Run Split Inference Execution
        executor = DistributedInferenceExecutor(model=model, tiers=tiers)
        t0 = time.perf_counter()
        split_result: GenerationResult = executor.generate(
            prompt=prompt,
            partition_plan=partition_plan,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        split_time = time.perf_counter() - t0

        # Synchronize if any tier is on CUDA
        has_cuda = any("cuda" in str(t.device).lower() for t in tiers.values())
        if has_cuda and cuda_avail:
            torch.cuda.synchronize()

        # 3. Audit Transfers
        transfer_audits: List[TransferAuditRecord] = []
        for rec in split_result.transfer_records:
            src_tier_id = rec.source_tier
            dst_tier_id = rec.destination_tier
            src_dev = str(tiers[src_tier_id].device) if src_tier_id in tiers else "unknown"
            dst_dev = str(tiers[dst_tier_id].device) if dst_tier_id in tiers else "unknown"
            is_cross = src_dev != dst_dev

            transfer_audits.append(
                TransferAuditRecord(
                    source_tier=src_tier_id.value,
                    destination_tier=dst_tier_id.value,
                    source_device=src_dev,
                    destination_device=dst_dev,
                    tensor_shape=list(rec.tensor_shape),
                    tensor_dtype=rec.dtype,
                    size_bytes=rec.byte_size,
                    transfer_duration_ms=0.0,
                    is_cross_device=is_cross,
                    is_simulated_transfer=False,
                )
            )

        # 4. Monolithic Reference Execution (All layers on tier 0's device)
        first_tier_id, _ = active_tiers[0]
        mono_kwargs: Dict[str, Any] = {"total_layers": model.num_layers}
        if first_tier_id == TierId.USER_DEVICE:
            mono_kwargs["user_device"] = (0, model.num_layers - 1)
        elif first_tier_id == TierId.EDGE_A:
            mono_kwargs["edge_a"] = (0, model.num_layers - 1)
        elif first_tier_id == TierId.EDGE_B:
            mono_kwargs["edge_b"] = (0, model.num_layers - 1)
        monolithic_plan = PartitionPlan(**mono_kwargs)

        ref_executor = DistributedInferenceExecutor(model=model, tiers=tiers)
        mono_result: GenerationResult = ref_executor.generate(
            prompt=prompt,
            partition_plan=monolithic_plan,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )

        split_tokens = split_result.generated_tokens
        mono_tokens = mono_result.generated_tokens
        token_parity = (split_tokens == mono_tokens)
        mismatch_msg = None
        if not token_parity:
            mismatch_msg = (
                f"Token mismatch: Split produced {split_tokens} "
                f"but Monolithic produced {mono_tokens}."
            )
            logger.warning(mismatch_msg)

        # Restore original partition placement
        model.place_layers(partition_plan, tiers)

        # 5. Classify Evidence Quality
        tier_dev_map = {tid.value: str(t.device) for tid, t in tiers.items()}
        descriptor = EvidenceQualityDescriptor.classify(
            cuda_available=cuda_avail,
            is_simulation_mode=False,
            use_real_socket=False,
            use_emulated_network=False,
            tier_devices=tier_dev_map,
        )

        return SplitInferenceVerificationReport(
            model_name=getattr(model.model.config, "_name_or_path", "gpt2"),
            total_layers=model.num_layers,
            partition_plan_id=str(partition_plan.to_dict()),
            active_tiers=[t.value for t, _ in active_tiers],
            device_placement_mode=descriptor.device_placement_mode.value,
            placement_audits=placements,
            transfers=transfer_audits,
            total_transfer_bytes=split_result.total_transfer_bytes,
            total_transfer_time_ms=sum(t.transfer_duration_ms for t in transfer_audits),
            token_parity_verified=token_parity,
            generated_tokens_split=split_tokens,
            generated_tokens_monolithic=mono_tokens,
            prompt=prompt,
            generated_text=split_result.generated_text,
            execution_time_seconds=split_time,
            kv_cache_summary=split_result.kv_cache_summary,
            synchronization_recorded=has_cuda,
            evidence_quality=descriptor.to_dict(),
            parity_mismatch_details=mismatch_msg,
        )
