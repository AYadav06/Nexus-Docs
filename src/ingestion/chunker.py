from src.core.config import settings
from src.core.models import DocumentChunk, DocumentMetadata


class RecursiveChunker:
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Priority order of split points: paragraph -> line -> sentence -> word -> character
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text by separators until pieces are under chunk_size."""
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits: list[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s[:self.chunk_size])
                else:
                    final_chunks.extend(self._split_text(s, new_separators))

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merges small splits together up to chunk_size, with chunk_overlap."""
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0

        for piece in splits:
            piece_len = len(piece) + (len(separator) if current_doc else 0)
            if total + piece_len > self.chunk_size:
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc.strip():
                        docs.append(doc.strip())

                    # Sliding overlap window
                    while total > self.chunk_overlap and current_doc:
                        removed = current_doc.pop(0)
                        total -= len(removed) + len(separator)
            current_doc.append(piece)
            total += piece_len

        if current_doc:
            doc = separator.join(current_doc)
            if doc.strip():
                docs.append(doc.strip())

        return docs

    def split_to_chunks(self, text: str, base_metadata: DocumentMetadata) -> list[DocumentChunk]:
        """Splits raw text and returns typed DocumentChunk objects with metadata."""
        raw_chunks = self._split_text(text, self.separators)
        chunks: list[DocumentChunk] = []
        for raw in raw_chunks:
            if not raw.strip():
                continue
            chunk = DocumentChunk.create(
                content=raw.strip(),
                metadata=base_metadata.model_copy()
            )
            chunks.append(chunk)
        return chunks
