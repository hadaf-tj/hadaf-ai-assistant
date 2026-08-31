"""RAG (Retrieval-Augmented Generation) package."""

from app.rag.embeddings import BaseEmbeddingModel, HashEmbeddingModel
from app.rag.loader import MarkdownDocumentLoader
from app.rag.models import Chunk, Document, SearchResult
from app.rag.pipeline import IngestionResult, RAGPipeline
from app.rag.splitter import MarkdownSectionSplitter
from app.rag.vector_store import BaseVectorStore, InMemoryVectorStore

__all__ = [
    "BaseEmbeddingModel",
    "BaseVectorStore",
    "Chunk",
    "Document",
    "HashEmbeddingModel",
    "InMemoryVectorStore",
    "IngestionResult",
    "MarkdownDocumentLoader",
    "MarkdownSectionSplitter",
    "RAGPipeline",
    "SearchResult",
]
