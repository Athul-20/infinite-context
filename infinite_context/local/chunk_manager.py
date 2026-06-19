"""
Token chunk manager for the local TTT engine.

Splits massive token sequences into fixed-size windows so they fit inside
the model's KV-Cache limits during the Test-Time Training reading phase.
"""

from typing import List

__all__ = ["ChunkManager"]


class ChunkManager:
    """Splits token lists into overlapping fixed-size chunks.

    Args:
        chunk_size: Number of tokens per chunk.
        overlap: Number of overlapping tokens between consecutive chunks
                 (helps preserve context continuity).
    """

    def __init__(self, chunk_size: int = 1024, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_tokens(self, tokens: List[int]) -> List[List[int]]:
        """Split *tokens* into smaller manageable chunks."""
        if not tokens:
            return []

        step = self.chunk_size - self.overlap

        # Ensure we always move forward.
        if step <= 0:
            raise ValueError("Overlap cannot be greater than or equal to chunk_size.")

        chunks = []
        for i in range(0, len(tokens), step):
            chunks.append(tokens[i : i + self.chunk_size])

        return chunks
