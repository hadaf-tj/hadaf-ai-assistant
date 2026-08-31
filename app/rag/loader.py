"""Markdown document loader for RAG ingestion."""

import os
from pathlib import Path
from typing import List, Optional, Union

from app.rag.models import Document


class MarkdownDocumentLoader:
    """Loader responsible for reading and parsing markdown documents."""

    def __init__(self, base_path: Optional[Union[str, Path]] = None) -> None:
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def _extract_title(self, content: str) -> Optional[str]:
        """Extract document title from the first H1 header if present."""
        for line in content.splitlines():
            line_str = line.strip()
            if line_str.startswith("# "):
                return line_str[2:].strip()
        return None

    def load_file(self, file_path: Union[str, Path]) -> Document:
        """Load a single markdown file into a Document object."""
        path = Path(file_path)
        if not path.is_absolute():
            path = (self.base_path / path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Specified path is not a file: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8-sig") as f:
                content = f.read()

        title = self._extract_title(content) or path.stem
        relative_source = str(path)
        try:
            relative_source = str(path.relative_to(self.base_path))
        except ValueError:
            pass

        return Document(
            id=path.stem,
            content=content,
            metadata={
                "source": str(path),
                "relative_source": relative_source,
                "file_name": path.name,
                "doc_title": title,
                "char_length": len(content),
                "line_count": len(content.splitlines()),
            },
        )

    def load_files(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """Load multiple markdown files."""
        documents: List[Document] = []
        for file_path in file_paths:
            documents.append(self.load_file(file_path))
        return documents

    def load_directory(
        self,
        dir_path: Optional[Union[str, Path]] = None,
        glob_pattern: str = "**/*.md",
    ) -> List[Document]:
        """Recursively load all markdown files from a directory."""
        target_dir = Path(dir_path) if dir_path else self.base_path
        if not target_dir.is_absolute():
            target_dir = (self.base_path / target_dir).resolve()

        if not target_dir.exists() or not target_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {target_dir}")

        documents: List[Document] = []
        for path in sorted(target_dir.glob(glob_pattern)):
            if path.is_file():
                documents.append(self.load_file(path))
        return documents
