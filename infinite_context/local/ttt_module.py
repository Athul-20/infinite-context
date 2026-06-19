"""
In-Place Test-Time Training (TTT) module.

Attaches transient LoRA fast-weight adapters to an LLM's MLP layers and
trains them on incoming context chunks so the model "memorises" the document
into its weights at inference time.
"""

import torch
import torch.nn as nn
from typing import List, Optional

__all__ = ["InPlaceTTT"]

# Lazy guard — resolved once at first use, not at import time.
_peft_available: Optional[bool] = None


def _ensure_peft():
    """Import peft on first use and raise a clear error if missing."""
    global _peft_available
    if _peft_available is None:
        try:
            import peft  # noqa: F401
            _peft_available = True
        except ImportError:
            _peft_available = False
    if not _peft_available:
        raise ImportError(
            "PEFT is required for the local engine. "
            "Install via:  pip install infinite_context[local]"
        )


class InPlaceTTT:
    """Manages LoRA adapter injection, test-time training, and teardown."""

    def __init__(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
        lr: float = 5e-3,
        adapter_name: str = "ttt_fast_weights",
    ):
        """
        Args:
            model: The Hugging Face PreTrainedModel.
            target_modules: MLP layer names to attach fast weights to.
                            Defaults to ``["down_proj"]``.
            lr: Learning rate for the test-time training update.
            adapter_name: Name of the PEFT adapter used for the fast weights.
        """
        _ensure_peft()

        self.model = model
        self.target_modules = target_modules or ["down_proj"]
        self.lr = lr
        self.adapter_name = adapter_name
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self._setup_lora()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _setup_lora(self) -> None:
        """Inject Fast Weights (LoRA) into the base model.

        By setting ``r=8`` and targeting ``down_proj``, we turn the static
        MLP into an active, online memory bank.
        """
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=self.target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=True,
        )
        self.model = get_peft_model(self.model, config, adapter_name=self.adapter_name)

        # Train ONLY the LoRA parameters — not the frozen base weights.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_on_chunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Run one gradient step on *input_ids* (the 'Reading Phase').

        Uses the standard causal next-token prediction loss which achieves
        the same goal as the paper's Conv1D LM-Aligned objective: baking
        the chunk's knowledge into the ``down_proj`` weights.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Mixed-precision forward pass (FP16 on CUDA, no-op on CPU).
        device_type = input_ids.device.type
        with torch.amp.autocast(device_type, enabled=(device_type == "cuda")):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss

        # Backward + update ONLY the LoRA adapters.
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def remove_fast_weights(self) -> None:
        """Delete the LoRA adapter ('State Destruction').

        The model instantly resets to its base pre-trained state, ready for
        the next request.
        """
        self.model.eval()
        self.model.delete_adapter(self.adapter_name)
