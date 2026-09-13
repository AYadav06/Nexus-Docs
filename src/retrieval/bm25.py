import re
from rank_bm25 import BM25Okapi
from src.core.models import DocumentChunk


class BM25Retriever:
    def __init__(self, chunks: list[DocumentChunk] | None = None):
        self.chunks: list[DocumentChunk] = []
        self.bm25: BM25Okapi | None = None
        if chunks:
            self.index_chunks(chunks)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text by words, code symbols, and numbers (lowercased)."""
        return re.findall(r"\w+", text.lower())

    def index_chunks(self, chunks: list[DocumentChunk]):
        """Builds the BM25 inverted index over the provided chunks."""
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        """Returns top_k matching chunks with their BM25 score."""
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Pair chunks with scores and sort descending
        scored_pairs = list(zip(self.chunks, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return scored_pairs[:top_k]
