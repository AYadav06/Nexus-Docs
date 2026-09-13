from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import hashlib


class DocumentType(str, Enum):
    PDF = "pdf"
    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"


class DocumentMetadata(BaseModel):
    source_path: str
    doc_type: DocumentType
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    section_title: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: DocumentMetadata

    @classmethod
    def create(cls, content: str, metadata: DocumentMetadata, chunk_index: int = 0) -> "DocumentChunk":
        """Generates a deterministic hash for deduplication."""
        unique_string = f"{metadata.source_path}:{metadata.page_number}:{chunk_index}:{content.strip()}"
        chunk_id = hashlib.sha256(unique_string.encode()).hexdigest()[:16]
        return cls(chunk_id=chunk_id, content=content, metadata=metadata)

