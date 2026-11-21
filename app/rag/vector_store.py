"""Vector store abstractions and in-memory implementation."""

import math
from abc import ABC, abstractmethod
from typing import Any

from app.rag.models import Chunk, SearchResult


class BaseVectorStore(ABC):
    """Abstract interface for vector database storage and similarity search."""

    @abstractmethod
    async def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        """Add chunks with embeddings to the vector store."""

    @abstractmethod
    async def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for top_k most similar chunks."""

    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a single chunk by ID."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored vectors."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored chunks."""


class InMemoryVectorStore(BaseVectorStore):
    """Lightweight in-memory vector store using cosine similarity."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        sim = dot / (norm_a * norm_b)
        return max(-1.0, min(1.0, sim))

    async def add_chunks(self, chunks: list[Chunk]) -> list[str]:
        added_ids: list[str] = []
        for chunk in chunks:
            if not chunk.embedding:
                raise ValueError(f"Chunk {chunk.id} is missing an embedding vector.")
            self._chunks[chunk.id] = chunk
            added_ids.append(chunk.id)
        return added_ids

    async def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        scored_results: list[SearchResult] = []

        for chunk in self._chunks.values():
            if not chunk.embedding:
                continue

            # Apply metadata filtering if specified
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            sim = self._cosine_similarity(query_vector, chunk.embedding)
            if sim >= min_score:
                scored_results.append(SearchResult(chunk=chunk, score=sim))

        # Sort descending by score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    async def clear(self) -> None:
        self._chunks.clear()

    def count(self) -> int:
        return len(self._chunks)
