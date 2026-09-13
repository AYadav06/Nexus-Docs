from collections import defaultdict
from src.core.models import DocumentChunk
from src.vectorstore.chroma import ChromaVectorStore
from src.retrieval.bm25 import BM25Retriever


class HybridRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 4, candidate_k: int = 20) -> list[DocumentChunk]:
        """
        Executes Dense (Vector) + Sparse (BM25) searches and fuses results via RRF.
        Returns top_k reranked DocumentChunks.
        """
        # 1. Fetch candidates from both search engines
        dense_results = self.vector_store.search(query, top_k=candidate_k)
        sparse_results = self.bm25_retriever.search(query, top_k=candidate_k)

        # 2. Reciprocal Rank Fusion (RRF)
        # RRF_Score(doc) = sum(1 / (k + rank))
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, DocumentChunk] = {}

        # Dense ranks (rank is 1-indexed)
        for rank, (chunk, _) in enumerate(dense_results, 1):
            rrf_scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)
            chunk_map[chunk.chunk_id] = chunk

        # Sparse ranks
        for rank, (chunk, _) in enumerate(sparse_results, 1):
            rrf_scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)
            chunk_map[chunk.chunk_id] = chunk

        # 3. Sort by fused RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        # 4. Return top_k best chunks
        return [chunk_map[cid] for cid in sorted_chunk_ids[:top_k]]
