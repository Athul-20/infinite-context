"""
Context Gateway — unified interface for the cloud and local engines.

Users interact exclusively through :class:`ContextGateway`, selecting either
``engine="cloud"`` (Headroom compression + API) or ``engine="local"``
(In-Place Test-Time Training on a local GPU).
"""

from typing import Any, Generator, List, Optional

__all__ = ["ContextGateway"]


class ContextGateway:
    """Hybrid gateway that routes queries through either a cloud API or a
    local TTT engine depending on configuration.

    Args:
        engine: ``"cloud"`` for Headroom API compression, or ``"local"``
                for In-Place TTT.
        model_id: Target model (e.g. ``"claude-3-5-sonnet-20240620"`` or a
                  local HuggingFace repo id).
        api_key: API key for cloud routing (if applicable).
        base_url: Custom base URL for OpenAI-compatible providers like Groq.
        compression_ratio: Target compression ratio for Headroom (cloud only).
        skip_compression: If *True*, skips Headroom compression (useful for
                          testing alternative APIs directly).
        **kwargs: Additional parameters forwarded to :class:`LocalEngine`
                  (e.g. ``device``, ``load_in_4bit``).
    """

    def __init__(
        self,
        engine: str = "cloud",
        model_id: str = "claude-3-5-sonnet-20240620",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        compression_ratio: float = 0.8,
        skip_compression: bool = False,
        **kwargs: Any,
    ):
        if engine not in ("cloud", "local"):
            raise ValueError("Engine must be either 'cloud' or 'local'.")

        self.engine = engine
        self.model_id = model_id

        if self.engine == "cloud":
            from .cloud.compressor import HeadroomCompressor
            from .cloud.client import CloudClient

            self.compressor = (
                HeadroomCompressor(target_ratio=compression_ratio)
                if not skip_compression
                else None
            )
            self.client = CloudClient(
                api_key=api_key, model_id=self.model_id, base_url=base_url,
            )
        else:  # local
            from .local.engine import LocalEngine
            self.local_engine = LocalEngine(model_id=self.model_id, **kwargs)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        question: str,
        context: str,
        history: Optional[List[dict]] = None,
        max_history: int = 4,
        keep_state: bool = False,
    ) -> str:
        """Process a massive context payload and answer the question.

        Includes a rolling-window history to keep token costs low.
        """
        if self.engine == "cloud":
            compressed = (
                self.compressor.compress(context)
                if self.compressor is not None
                else context
            )

            messages: list[dict] = [{"role": "system", "content": compressed}]
            if history:
                messages.extend(history[-max_history:])
            messages.append({"role": "user", "content": question})

            return self.client.send_request(messages)

        # local
        return self.local_engine.generate(
            question, context, keep_state=keep_state,
        )

    def stream_chat(
        self,
        question: str,
        context: str,
        history: Optional[List[dict]] = None,
        max_history: int = 4,
    ) -> Generator[str, None, None]:
        """Yield response tokens as a stream (cloud engine only)."""
        if self.engine == "cloud":
            compressed = (
                self.compressor.compress(context)
                if self.compressor is not None
                else context
            )

            messages: list[dict] = [{"role": "system", "content": compressed}]
            if history:
                messages.extend(history[-max_history:])
            messages.append({"role": "user", "content": question})

            yield from self.client.stream_request(messages)
        else:
            raise NotImplementedError(
                "Streaming is not supported by the local TTT engine."
            )

    # ------------------------------------------------------------------
    # Persistence (local engine only)
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Save trained Fast Weights to disk for later resumption."""
        if self.engine == "local":
            self.local_engine.save_state(path)
        else:
            raise NotImplementedError(
                "Cloud engine does not support local state saving."
            )

    def load_state(self, path: str) -> None:
        """Reload previously-saved Fast Weights from disk."""
        if self.engine == "local":
            self.local_engine.load_state(path)
        else:
            raise NotImplementedError(
                "Cloud engine does not support local state loading."
            )
