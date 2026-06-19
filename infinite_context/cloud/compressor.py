"""
Headroom semantic compression wrapper.

Compresses massive context payloads locally before sending them to a cloud
LLM, reducing token counts and API costs.
"""

__all__ = ["HeadroomCompressor"]


class HeadroomCompressor:
    """Wraps the ``headroom-ai`` library with a graceful fallback."""

    def __init__(self, target_ratio: float = 0.8):
        self.target_ratio = target_ratio
        self._compress_fn = None

        try:
            from headroom import compress
            self._compress_fn = compress
        except ImportError:
            print(
                "Warning: headroom-ai not installed. "
                "Using fallback semantic compression simulator."
            )

    def compress(self, context: str) -> str:
        """Compress *context* using Headroom's algorithms.

        Falls back to a head + tail truncation when the library is absent.
        """
        if self._compress_fn is not None:
            return self._compress_fn(context)

        # Fallback simulator: keep the first 1000 chars and the last 2000
        # chars (where needles usually are).
        if len(context) < 3000:
            return context

        return (
            context[:1000]
            + "\n\n...[CONTENT COMPRESSED BY HEADROOM SIMULATOR]...\n\n"
            + context[-2000:]
        )
