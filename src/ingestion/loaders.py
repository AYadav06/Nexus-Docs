from abc import ABC, abstractmethod
from pathlib import Path
from pypdf import PdfReader
from src.core.models import DocumentChunk, DocumentMetadata, DocumentType
from src.ingestion.chunker import RecursiveChunker


class BaseLoader(ABC):
    def __init__(self, chunker: RecursiveChunker | None = None):
        self.chunker = chunker or RecursiveChunker()

    @abstractmethod
    def load(self, file_path: Path) -> list[DocumentChunk]:
        """Extract and chunk the document into DocumentChunk objects."""
        pass


class PDFLoader(BaseLoader):
    def load(self, file_path: Path) -> list[DocumentChunk]:
        reader = PdfReader(str(file_path))
        chunks: list[DocumentChunk] = []

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            metadata = DocumentMetadata(
                source_path=str(file_path),
                doc_type=DocumentType.PDF,
                page_number=page_idx + 1,
            )
            page_chunks = self.chunker.split_to_chunks(text, metadata)
            chunks.extend(page_chunks)

        return chunks


class MarkdownLoader(BaseLoader):
    def load(self, file_path: Path) -> list[DocumentChunk]:
        content = file_path.read_text(encoding="utf-8")
        metadata = DocumentMetadata(
            source_path=str(file_path),
            doc_type=DocumentType.MARKDOWN,
        )
        return self.chunker.split_to_chunks(content, metadata)


class CodeLoader(BaseLoader):
    def load(self, file_path: Path) -> list[DocumentChunk]:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # We chunk code with line numbers tracking
        metadata = DocumentMetadata(
            source_path=str(file_path),
            doc_type=DocumentType.CODE,
            line_start=1,
            line_end=len(lines),
            extra={"extension": file_path.suffix}
        )
        return self.chunker.split_to_chunks(content, metadata)
