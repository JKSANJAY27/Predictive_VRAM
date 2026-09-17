"""
Model layer decomposition and partitioned execution for autoregressive Transformers.

Provides modular layer access, embeddings, block-range forward execution,
and output projection for split inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from src.runtime.partition import PartitionPlan
from src.runtime.tier import Tier, TierId


class LayeredTransformer(nn.Module):
    """
    Wraps an autoregressive Transformer to expose fine-grained layer-by-layer execution,
    allowing arbitrary layer placement across distributed or logical execution tiers.
    """

    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: Any,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.base_device = device

        # Deconstruct GPT2 components
        self.wte = self.model.transformer.wte
        self.wpe = self.model.transformer.wpe
        self.drop = self.model.transformer.drop
        self.blocks = self.model.transformer.h
        self.ln_f = self.model.transformer.ln_f
        self.lm_head = self.model.lm_head

        self.num_layers = len(self.blocks)
        self.hidden_size = self.model.config.n_embd
        self.num_heads = self.model.config.n_head

        # Ensure all attention modules have layer_idx explicitly configured
        for idx, block in enumerate(self.blocks):
            if hasattr(block, "attn") and getattr(block.attn, "layer_idx", None) is None:
                block.attn.layer_idx = idx

    def embed(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_length: int = 0,
    ) -> torch.Tensor:
        """
        Compute initial input embeddings (word token embeddings + position embeddings).

        Args:
            input_ids: Token ID tensor of shape (batch_size, seq_len).
            position_ids: Optional explicit position IDs.
            past_length: Sequence length offset from preceding cached tokens.

        Returns:
            Hidden states tensor of shape (batch_size, seq_len, hidden_size).
        """
        device = input_ids.device
        seq_len = input_ids.shape[1]

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + seq_len,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        return hidden_states

    def forward_layer_range(
        self,
        hidden_states: torch.Tensor,
        start_layer: int,
        end_layer: int,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Execute a contiguous slice of transformer blocks [start_layer, end_layer] inclusive.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            start_layer: Inclusive starting block index.
            end_layer: Inclusive ending block index.
            past_key_values: Optional DynamicCache instance updated in-place.
            attention_mask: Optional attention mask.

        Returns:
            Output hidden states tensor from the final layer in the range.
        """
        if start_layer < 0 or end_layer >= self.num_layers or start_layer > end_layer:
            raise ValueError(
                f"Invalid layer range [{start_layer}, {end_layer}] for model with {self.num_layers} layers."
            )

        for layer_idx in range(start_layer, end_layer + 1):
            block = self.blocks[layer_idx]
            block_device = next(block.parameters()).device

            # Ensure input hidden state is on the layer's device
            if hidden_states.device != block_device:
                hidden_states = hidden_states.to(block_device)

            output = block(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )
            hidden_states = output[0] if isinstance(output, tuple) else output

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply final layer normalization and language modeling head projection.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        ln_device = next(self.ln_f.parameters()).device
        if hidden_states.device != ln_device:
            hidden_states = hidden_states.to(ln_device)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def place_layers(
        self,
        partition_plan: PartitionPlan,
        tiers: Dict[TierId, Tier],
    ) -> None:
        """
        Place model layers, embeddings, and output heads on the devices specified
        by the active tiers in the partition plan.
        """
        active_tiers = partition_plan.get_active_tiers()
        if not active_tiers:
            raise ValueError("Cannot place layers: PartitionPlan has no active tiers.")

        first_tier_id, _ = active_tiers[0]
        last_tier_id, _ = active_tiers[-1]

        first_device = tiers[first_tier_id].device
        last_device = tiers[last_tier_id].device

        # Embeddings go to the first tier
        self.wte.to(first_device)
        self.wpe.to(first_device)
        self.drop.to(first_device)

        # Transformer blocks to their respective tiers
        for tier_id, (start, end) in active_tiers:
            tier_dev = tiers[tier_id].device
            for idx in range(start, end + 1):
                self.blocks[idx].to(tier_dev)

        # Final norm and LM head go to the last tier
        self.ln_f.to(last_device)
        self.lm_head.to(last_device)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "gpt2",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> LayeredTransformer:
        """Load pretrained model and tokenizer from HuggingFace."""
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
        )
        model.to(device)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, device=device)

    @classmethod
    def create_synthetic(
        cls,
        num_layers: int = 4,
        hidden_size: int = 64,
        num_heads: int = 2,
        vocab_size: int = 1000,
        device: str = "cpu",
    ) -> LayeredTransformer:
        """
        Create a lightweight synthetic model for fast, offline unit testing.
        """
        config = GPT2Config(
            n_layer=num_layers,
            n_embd=hidden_size,
            n_head=num_heads,
            vocab_size=vocab_size,
            n_positions=256,
            n_ctx=256,
        )
        model = GPT2LMHeadModel(config)
        for idx, block in enumerate(model.transformer.h):
            block.attn.layer_idx = idx
        model.to(device)
        model.eval()

        # Dummy tokenizer wrapper for synthetic model testing
        class DummyTokenizer:
            eos_token_id = 0
            pad_token_id = 0

            def __call__(self, text: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
                # Simple deterministic ascii-based tokens for test reproducibility
                tokens = [min(ord(c), vocab_size - 1) for c in text][:32] or [1]
                return {
                    "input_ids": torch.tensor([tokens], dtype=torch.long, device=device),
                    "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long, device=device),
                }

            def encode(self, text: str, return_tensors: Optional[str] = None) -> Any:
                out = self(text, return_tensors=return_tensors or "pt")["input_ids"]
                if return_tensors is None:
                    return out[0].tolist()
                return out

            def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[0]
                return "".join(chr(t) if 32 <= t <= 126 else f"<{t}>" for t in token_ids)

        return cls(model=model, tokenizer=DummyTokenizer(), device=device)
