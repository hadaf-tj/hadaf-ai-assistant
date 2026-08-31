"""Section-aware Markdown document splitter."""

import re
from typing import List, Optional, Tuple

from app.rag.models import Chunk, Document


class MarkdownSectionSplitter:
    """Splits markdown documents based on header hierarchy and paragraph boundaries."""

    def __init__(
        self,
        max_chunk_size: int = 800,
        chunk_overlap: int = 100,
        preserve_headers_in_content: bool = True,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_headers_in_content = preserve_headers_in_content
        # Regex for matching markdown headers (# through ####)
        self.header_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    def _split_into_sections(self, text: str) -> List[Tuple[List[str], str]]:
        """Split markdown text into logical sections tracking the header hierarchy."""
        lines = text.splitlines()
        sections: List[Tuple[List[str], str]] = []
        current_hierarchy: List[str] = []
        current_lines: List[str] = []

        for line in lines:
            header_match = self.header_pattern.match(line)
            if header_match:
                # Flush previous section
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        sections.append((list(current_hierarchy), content))
                    current_lines = []

                hashes, title = header_match.groups()
                level = len(hashes)
                title = title.strip()

                # Adjust hierarchy depth
                current_hierarchy = current_hierarchy[: level - 1]
                current_hierarchy.append(title)
                current_lines.append(line)
            else:
                current_lines.append(line)

        # Flush trailing section
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((list(current_hierarchy), content))

        return sections

    def _split_long_text(self, text: str) -> List[str]:
        """Split text exceeding max_chunk_size by paragraphs or sentences with overlap."""
        if len(text) <= self.max_chunk_size:
            return [text]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            if not current_chunk:
                current_chunk = para
            elif len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                current_chunk += "\n\n" + para
            else:
                chunks.append(current_chunk)
                # Apply overlap by taking trailing portion if available
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    overlap_seed = current_chunk[-self.chunk_overlap :]
                    current_chunk = overlap_seed + "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        # If a single paragraph was longer than max_chunk_size, chunk by characters/lines
        final_chunks: List[str] = []
        for c in chunks:
            if len(c) <= self.max_chunk_size:
                final_chunks.append(c)
            else:
                start = 0
                step = max(self.max_chunk_size - self.chunk_overlap, 100)
                while start < len(c):
                    end = min(start + self.max_chunk_size, len(c))
                    sub_chunk = c[start:end].strip()
                    if sub_chunk:
                        final_chunks.append(sub_chunk)
                    start += step

        return final_chunks

    def split_document(self, document: Document) -> List[Chunk]:
        """Split a Document into a list of Chunk objects."""
        sections = self._split_into_sections(document.content)
        chunks: List[Chunk] = []
        chunk_idx = 0

        # If no sections were found (e.g. no markdown headers), treat entire doc as one section
        if not sections:
            doc_title = document.metadata.get("doc_title") or document.id
            sections = [([doc_title], document.content)]

        for hierarchy, section_text in sections:
            section_title = hierarchy[-1] if hierarchy else (document.metadata.get("doc_title") or "")
            sub_texts = self._split_long_text(section_text)

            for sub_text in sub_texts:
                if not sub_text.strip():
                    continue

                chunk_id = f"{document.id}_chunk_{chunk_idx:03d}"
                breadcrumbs = " > ".join(hierarchy) if hierarchy else section_title

                # Contextual prefix for rich semantic embeddings and retrieval
                content_for_chunk = sub_text
                if self.preserve_headers_in_content:
                    file_name = document.metadata.get("file_name") or document.id
                    if breadcrumbs:
                        breadcrumb_str = f"{file_name} > {breadcrumbs}" if (file_name and file_name not in breadcrumbs) else breadcrumbs
                    else:
                        breadcrumb_str = str(file_name) if file_name else ""

                    if breadcrumb_str:
                        prefix = f"[{breadcrumb_str}]\n"
                        if not sub_text.startswith(prefix):
                            content_for_chunk = f"{prefix}{sub_text}"

                metadata = {
                    **document.metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "section_title": section_title,
                    "section_hierarchy": hierarchy,
                    "breadcrumbs": breadcrumbs,
                    "char_count": len(content_for_chunk),
                }

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        content=content_for_chunk,
                        metadata=metadata,
                    )
                )
                chunk_idx += 1

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split multiple documents into chunks."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks
