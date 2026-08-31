"""RAG data models and schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Raw loaded document."""

    id: str = Field(description="Unique identifier of the document")
    content: str = Field(description="Raw text content of the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata including source path, title, etc.",
    )


class Chunk(BaseModel):
    """Processed document chunk ready for embedding and indexing."""

    id: str = Field(description="Unique identifier of the chunk")
    content: str = Field(description="Chunk text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata including source, section, breadcrumbs",
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector representation of the chunk content",
    )


class SearchResult(BaseModel):
    """Result returned from a vector similarity search."""

    chunk: Chunk = Field(description="Retrieved chunk")
    score: float = Field(description="Similarity score (higher is more relevant)")
