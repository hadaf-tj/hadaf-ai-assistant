"""Unit tests for MarkdownDocumentLoader."""

import pytest
from pathlib import Path
from app.rag.loader import MarkdownDocumentLoader


def test_loader_single_file(tmp_path: Path):
    """Test loading a single markdown document."""
    file_path = tmp_path / "test_doc.md"
    file_path.write_text("# Test Title\n\nThis is test content.", encoding="utf-8")

    loader = MarkdownDocumentLoader(base_path=tmp_path)
    doc = loader.load_file(file_path)

    assert doc.id == "test_doc"
    assert "This is test content." in doc.content
    assert doc.metadata["doc_title"] == "Test Title"
    assert doc.metadata["file_name"] == "test_doc.md"
    assert doc.metadata["line_count"] == 3


def test_loader_directory(tmp_path: Path):
    """Test loading all markdown files from a directory."""
    (tmp_path / "doc1.md").write_text("# Doc 1\nContent 1", encoding="utf-8")
    (tmp_path / "doc2.md").write_text("# Doc 2\nContent 2", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("Ignored", encoding="utf-8")

    loader = MarkdownDocumentLoader(base_path=tmp_path)
    docs = loader.load_directory(tmp_path)

    assert len(docs) == 2
    titles = {d.metadata["doc_title"] for d in docs}
    assert titles == {"Doc 1", "Doc 2"}


def test_loader_nonexistent_file(tmp_path: Path):
    """Test error handling when loading a missing file."""
    loader = MarkdownDocumentLoader(base_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_file("missing_file.md")
