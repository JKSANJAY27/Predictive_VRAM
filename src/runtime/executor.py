"""
Distributed split-inference executor for autoregressive Transformers.

Orchestrates forward execution across execution tiers, coordinates inter-tier
tensor transfers, maintains KV-cache state, and generates output tokens.

Module 2 extension:
    The generate() method accepts an optional `telemetry_buffer` parameter.
    When supplied, system telemetry (memory, CPU, activations, KV-cache,
    network conditions) is collected at every autoregressive step and pushed
    to the buffer. When None (the default), behaviour is identical to Module 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from src.runtime.model import LayeredTransformer
from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId
from src.runtime.transfer import TransferManager, TransferRecord

# Telemetry imports — guarded so Module 1 code paths remain unaffected
if TYPE_CHECKING:
    from src.telemetry.buffer import TelemetryBuffer


@dataclass
class GenerationResult:
    """
    Structured outcome of an autoregressive inference run.

    Module 2 additions (backward-compatible):
        telemetry_snapshot_count: Number of TelemetrySnapshots collected during
            this run (0 when telemetry_buffer was not supplied).
        telemetry_buffer: Reference to the live TelemetryBuffer used during this
            run, or None if telemetry was disabled.
    """
    prompt: str
    generated_text: str
    full_text: str
    prompt_tokens: List[int]
    generated_tokens: List[int]
    prompt_token_count: int
    generated_token_count: int
    partition_plan: PartitionPlan
    tiers: Dict[str, Dict[str, Any]]
    execution_time_seconds: float
    time_to_first_token_seconds: float
    inter_token_latencies: List[float]
    transfer_records: List[TransferRecord]
    total_transfers: int
    total_transfer_bytes: int
    kv_cache_summary: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Module 2 — telemetry fields (default values preserve backward compatibility)
    telemetry_snapshot_count: int = 0
    telemetry_buffer: Optional[Any] = None  # Optional[TelemetryBuffer]

    @property
    def tokens_per_second(self) -> float:
        if self.execution_time_seconds <= 0 or self.generated_token_count <= 0:
            return 0.0
        return self.generated_token_count / self.execution_time_seconds

    @property
    def average_token_latency_seconds(self) -> float:
        if not self.inter_token_latencies:
            return 0.0
        return sum(self.inter_token_latencies) / len(self.inter_token_latencies)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "full_text": self.full_text,
            "prompt_token_count": self.prompt_token_count,
            "generated_token_count": self.generated_token_count,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "time_to_first_token_seconds": round(self.time_to_first_token_seconds, 4),
            "average_token_latency_seconds": round(self.average_token_latency_seconds, 4),
            "partition_plan": self.partition_plan.to_dict(),
            "tiers": self.tiers,
            "total_transfers": self.total_transfers,
            "total_transfer_bytes": self.total_transfer_bytes,
            "kv_cache_summary": self.kv_cache_summary,
            "metadata": self.metadata,
            # Module 2
            "telemetry_snapshot_count": self.telemetry_snapshot_count,
        }


class DistributedInferenceExecutor:
    """
    Executes autoregressive generation across partitioned execution tiers.
    """

    def __init__(
        self,
        model: LayeredTransformer,
        tiers: Optional[Dict[TierId, Tier]] = None,
        transfer_manager: Optional[TransferManager] = None,
    ) -> None:
        self.model = model
        self.transfer_manager = transfer_manager or TransferManager()

        # Default tier topology if not supplied
        if tiers is None:
            self.tiers = {
                TierId.USER_DEVICE: Tier(tier_id=TierId.USER_DEVICE, name="User Device", device="cpu"),
                TierId.EDGE_A: Tier(tier_id=TierId.EDGE_A, name="Edge Node A", device="cpu"),
                TierId.EDGE_B: Tier(tier_id=TierId.EDGE_B, name="Edge Node B", device="cpu"),
            }
        else:
            self.tiers = tiers

    def _setup_tiers_for_plan(self, partition_plan: PartitionPlan) -> None:
        """Update tier layer ranges and place model layers according to partition plan."""
        for tier_id, tier in self.tiers.items():
            tier.layer_range = partition_plan.get_tier_range(tier_id)

        self.model.place_layers(partition_plan, self.tiers)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        partition_plan: PartitionPlan,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        telemetry_buffer: Optional[Any] = None,  # Optional[TelemetryBuffer]
    ) -> GenerationResult:
        """
        Execute split autoregressive generation for the given prompt under the partition plan.

        Args:
            prompt: Text prompt string.
            partition_plan: Validated PartitionPlan.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic greedy).
            do_sample: Whether to use sampling (greedy if False).
            eos_token_id: Optional stopping token ID.
            telemetry_buffer: Optional TelemetryBuffer. When provided, a
                TelemetrySnapshot is collected and pushed at every generation
                step. When None (default), no telemetry overhead is incurred
                and behaviour is identical to Module 1.

        Returns:
            GenerationResult containing generated text, timings, transfer
            telemetry, and (if enabled) a reference to the telemetry buffer.
        """
        # Lazily import telemetry components only when a buffer is supplied
        _telemetry_enabled = telemetry_buffer is not None
        if _telemetry_enabled:
            from src.telemetry.collectors import (
                MemoryCollector,
                CPUCollector,
                ActivationCollector,
                KVCacheCollector,
                NetworkConditionCollector,
            )
            from src.telemetry.types import TelemetrySnapshot
            _mem_collector  = MemoryCollector()
            _cpu_collector  = CPUCollector()
            _act_collector  = ActivationCollector()
            _kv_collector   = KVCacheCollector()
            _net_collector  = NetworkConditionCollector()
        start_wall_time = time.time()
        self.transfer_manager.clear_history()
        self._setup_tiers_for_plan(partition_plan)

        active_tiers = partition_plan.get_active_tiers()
        first_tier_id, _ = active_tiers[0]
        first_device = self.tiers[first_tier_id].device

        # 1. Tokenize prompt
        inputs = self.model.tokenizer(prompt, return_tensors="pt")
        input_ids: torch.Tensor = inputs["input_ids"].to(first_device)
        attention_mask: Optional[torch.Tensor] = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(first_device)

        prompt_tokens: List[int] = input_ids[0].tolist()
        prompt_token_count = len(prompt_tokens)
        generated_tokens: List[int] = []

        if eos_token_id is None:
            eos_token_id = getattr(self.model.tokenizer, "eos_token_id", None)

        from transformers import DynamicCache
        kv_cache = DynamicCache()

        inter_token_latencies: List[float] = []
        ttft: float = 0.0

        current_input_ids = input_ids
        past_length = 0

        # Autoregressive generation loop
        for step in range(max_new_tokens):
            step_start_time = time.time()

            # --- Telemetry: pre-step memory + CPU snapshot ---
            _step_mem_snap = None
            _step_cpu_snap = None
            _step_act_snaps: List[Any] = []
            if _telemetry_enabled:
                _step_mem_snap = _mem_collector.collect()
                _step_cpu_snap = _cpu_collector.collect()

            # A. Embedding (at the first tier)
            hidden_states = self.model.embed(
                current_input_ids,
                past_length=past_length,
            )

            # B. Forward through tiers in sequential order
            for i, (tier_id, (start_layer, end_layer)) in enumerate(active_tiers):
                # Inter-tier transfer if moving across boundaries
                if i > 0:
                    prev_tier_id, _ = active_tiers[i - 1]
                    target_device = self.tiers[tier_id].device

                    # --- Telemetry: capture activation before transfer ---
                    if _telemetry_enabled:
                        act_snap = _act_collector.collect(
                            tensor=hidden_states,
                            source_tier=prev_tier_id.value,
                            destination_tier=tier_id.value,
                            step=step,
                        )
                        _step_act_snaps.append(act_snap)

                    hidden_states = self.transfer_manager.transfer(
                        tensor=hidden_states,
                        source_tier=prev_tier_id,
                        destination_tier=tier_id,
                        target_device=target_device,
                        step=step,
                    )

                # Prepare mask for prefill vs single token
                tier_mask = None
                if step == 0 and attention_mask is not None:
                    if (attention_mask == 1).all():
                        tier_mask = None
                    else:
                        tier_mask = attention_mask.bool().to(self.tiers[tier_id].device)

                hidden_states = self.model.forward_layer_range(
                    hidden_states=hidden_states,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    past_key_values=kv_cache,
                    attention_mask=tier_mask,
                )

            # C. Compute logits at the final tier
            logits = self.model.compute_logits(hidden_states)
            next_token_logits = logits[:, -1, :]

            # D. Selection / Sampling
            if do_sample and temperature > 0.0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            next_token_int = int(next_token.item())
            generated_tokens.append(next_token_int)

            step_duration = time.time() - step_start_time
            if step == 0:
                ttft = step_duration
            else:
                inter_token_latencies.append(step_duration)

            # --- Telemetry: post-step KV-cache + network + push snapshot ---
            if _telemetry_enabled:
                _step_kv_snap  = _kv_collector.collect(kv_cache, step=step)
                _step_net_snap = _net_collector.collect()
                _composite = TelemetrySnapshot(
                    step=step,
                    timestamp=time.monotonic(),
                    memory=_step_mem_snap,
                    cpu=_step_cpu_snap,
                    kv_cache=_step_kv_snap,
                    network=_step_net_snap,
                    activations=list(_step_act_snaps),
                    metadata={"step_duration_s": step_duration},
                )
                telemetry_buffer.push(_composite)

            if eos_token_id is not None and next_token_int == eos_token_id:
                break

            # Prepare input for next decode step
            current_input_ids = next_token.to(first_device)
            past_length = prompt_token_count + len(generated_tokens) - 1

        total_exec_time = time.time() - start_wall_time

        # Decode generated tokens
        generated_text = self.model.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        full_text = prompt + generated_text

        # Compute KV-cache memory summary
        cached_layers_count = len(kv_cache.layers)
        total_kv_bytes = 0
        sample_layer_shape: List[int] = []
        for l_idx, layer_cache in enumerate(kv_cache.layers):
            if hasattr(layer_cache, "keys") and layer_cache.keys is not None:
                k = layer_cache.keys
                v = layer_cache.values
                total_kv_bytes += (k.numel() * k.element_size()) + (v.numel() * v.element_size())
                if not sample_layer_shape:
                    sample_layer_shape = list(k.shape)

        kv_summary = {
            "cached_layers_count": cached_layers_count,
            "total_kv_bytes": total_kv_bytes,
            "total_kv_mb": round(total_kv_bytes / (1024 * 1024), 4),
            "sample_layer_kv_shape": sample_layer_shape,
        }

        tier_summary = {
            tier_id.value: tier.to_dict() for tier_id, tier in self.tiers.items()
        }

        return GenerationResult(
            prompt=prompt,
            generated_text=generated_text,
            full_text=full_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            prompt_token_count=prompt_token_count,
            generated_token_count=len(generated_tokens),
            partition_plan=partition_plan,
            tiers=tier_summary,
            execution_time_seconds=total_exec_time,
            time_to_first_token_seconds=ttft,
            inter_token_latencies=inter_token_latencies,
            transfer_records=self.transfer_manager.get_history(),
            total_transfers=self.transfer_manager.total_transfers,
            total_transfer_bytes=self.transfer_manager.total_bytes,
            kv_cache_summary=kv_summary,
            # Module 2 telemetry
            telemetry_snapshot_count=(
                len(telemetry_buffer) if telemetry_buffer is not None else 0
            ),
            telemetry_buffer=telemetry_buffer,
        )
