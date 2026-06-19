"""
Local Test-Time Training (TTT) engine.

Orchestrates the full pipeline: model loading → chunking → LoRA injection →
reading phase (training) → answering phase (inference) → state management.
"""

import os
import torch
from typing import Any

from .chunk_manager import ChunkManager
from .ttt_module import InPlaceTTT

__all__ = ["LocalEngine"]

# Lazy import guard — resolved once at first use, not at import time.
_transformers_available: bool | None = None


def _ensure_transformers():
    """Import transformers on first use and raise a clear error if missing."""
    global _transformers_available
    if _transformers_available is None:
        try:
            import transformers  # noqa: F401
            _transformers_available = True
        except ImportError:
            _transformers_available = False
    if not _transformers_available:
        raise ImportError(
            "Transformers is required for the local engine. "
            "Install via:  pip install infinite_context[local]"
        )


class LocalEngine:
    """Orchestrates the Local Test-Time Training (TTT) engine.

    Loads the model in constrained memory (4-bit) and manages the Reading
    and Answering phases.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        load_in_4bit: bool = True,
        **kwargs: Any,
    ):
        _ensure_transformers()
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_id = model_id

        print(f"Loading base model {model_id} via Transformers into {self.device}...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        quantization_kwargs: dict = {}
        if load_in_4bit and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                quantization_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto" if self.device == "cuda" else None,
            **quantization_kwargs,
        )

        self.chunker = ChunkManager(chunk_size=1024, overlap=128)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        context: str,
        epochs_per_chunk: int = 15,
        target_loss: float = 0.05,
        keep_state: bool = False,
    ) -> str:
        """Run the full In-Place TTT pipeline.

        1. Break massive context into chunks.
        2. Inject Fast Weights (LoRA) into the model.
        3. Reading Phase — train the Fast Weights on the chunks.
        4. Answering Phase — generate the response to the question.
        5. State Management — optionally retain or destroy Fast Weights.
        """
        # 1. Chunking
        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        chunks = self.chunker.chunk_tokens(tokens)

        print(f"Bypassing KV-Cache: Splitting {len(tokens)} tokens into {len(chunks)} chunks.")

        # 2. Attach TTT Fast Weights (skip if a checkpoint was loaded)
        is_peft = hasattr(self.model, "peft_config")
        ttt: InPlaceTTT | None = None
        if not is_peft:
            print("Injecting TTT Fast Weights into down_proj layers...")
            ttt = InPlaceTTT(self.model)
        else:
            print("Model already has Fast Weights loaded from disk. Reusing them...")

        # 3. The 'Reading' Phase
        if chunks:
            print("Starting Reading Phase (Test-Time Training)...")
            if ttt is None:
                print("Warning: Cannot train loaded Fast Weights in-place. Skipping training.")
            else:
                for i, chunk in enumerate(chunks):
                    # Hoist tensor creation outside the epoch loop.
                    input_ids = torch.tensor([chunk], device=self.device)
                    for epoch in range(epochs_per_chunk):
                        loss = ttt.train_on_chunk(input_ids)
                        if loss < target_loss:
                            break
                    print(f"  Processed Chunk {i + 1}/{len(chunks)} - Epochs: {epoch + 1} - Loss: {loss:.4f}")

        # KV Cache is implicitly cleared here because we are not passing
        # past_key_values!

        # 4. The 'Answering' Phase
        print("Starting Answering Phase...")
        self.model.eval()

        prompt = f"Based on the codebase I just showed you, answer this question:\n{question}\nAnswer:"
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        device_type = self.device if self.device in ("cuda", "cpu") else "cpu"
        with torch.no_grad():
            output_ids = self.model.generate(
                prompt_ids,
                max_new_tokens=75,
                do_sample=False,
                repetition_penalty=1.1,
            )

        # Extract just the newly generated tokens.
        response_tokens = output_ids[0][prompt_ids.shape[1]:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        # 5. State Management
        if not keep_state:
            if ttt is not None:
                print("Resetting model state (deleting Fast Weights)...")
                ttt.remove_fast_weights()
            else:
                print("Cannot reset loaded Fast Weights. Retaining state.")
        else:
            print("Keeping model state (Fast Weights retained)...")
            if ttt is not None:
                # Update the reference so save_pretrained saves the adapter.
                self.model = ttt.model

        return response

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Save the current Fast Weights (LoRA adapter) to disk."""
        if hasattr(self.model, "peft_config"):
            self.model.save_pretrained(path)
            print(f"State saved to {path}")
        else:
            raise RuntimeError(
                "No active Fast Weights to save. "
                "Did you forget keep_state=True when calling generate()?"
            )

    def load_state(self, path: str) -> None:
        """Load Fast Weights (LoRA adapter) from disk."""
        from peft import PeftModel

        adapter_path = os.path.join(path, "ttt_fast_weights")
        if not os.path.exists(adapter_path):
            # Fallback for standard PEFT saves.
            adapter_path = path
        self.model = PeftModel.from_pretrained(
            self.model, adapter_path, is_trainable=True,
        )
        print(f"State loaded from {adapter_path}")
