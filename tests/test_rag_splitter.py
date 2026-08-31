"""Unit tests for MarkdownSectionSplitter."""

from app.rag.models import Document
from app.rag.splitter import MarkdownSectionSplitter


def test_splitter_sections_and_hierarchy():
    """Test splitting markdown with hierarchical headings."""
    content = """# Main Title

Intro paragraph.

## Section 1

Details for section 1.

### Subsection 1.1

Deep nested details.

## Section 2

Final notes.
"""
    doc = Document(
        id="sample",
        content=content,
        metadata={"source": "sample.md", "doc_title": "Main Title"},
    )

    splitter = MarkdownSectionSplitter(max_chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_document(doc)

    assert len(chunks) >= 4
    # Verify metadata on deep chunk
    deep_chunk = [c for c in chunks if "Deep nested details" in c.content][0]
    assert deep_chunk.metadata["section_title"] == "Subsection 1.1"
    assert deep_chunk.metadata["section_hierarchy"] == [
        "Main Title",
        "Section 1",
        "Subsection 1.1",
    ]
    assert deep_chunk.metadata["breadcrumbs"] == "Main Title > Section 1 > Subsection 1.1"


def test_splitter_long_paragraph():
    """Test that long sections exceeding max_chunk_size are sub-chunked."""
    long_para = "This is a long sentence repeated multiple times to exceed limits. " * 30
    content = f"# Header\n\n{long_para}"
    doc = Document(
        id="long_doc",
        content=content,
        metadata={"source": "long.md", "doc_title": "Header"},
    )

    splitter = MarkdownSectionSplitter(max_chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        # Check size constraints
        assert len(chunk.content) <= 500
