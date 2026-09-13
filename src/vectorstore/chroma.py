from pathlib import Path
import chromadb
from src.core.config import settings
from src.core.models import DocumentChunk, DocumentMetadata, DocumentType
from src.embeddings.gemini import GeminiEmbedder


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = "nexus_docs",
        persist_dir: Path = settings.VECTOR_DB_DIR,
        embedder: GeminiEmbedder | None = None,
    ):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize persistent on-disk client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedder = embedder or GeminiEmbedder()

        # Cosine distance collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embeds and upserts DocumentChunks. Skips already existing chunks via chunk_id."""
        if not chunks:
            return 0

        # Defensive deduplication by chunk_id to prevent ChromaDB DuplicateIDError
        unique_chunks_map = {c.chunk_id: c for c in chunks}
        chunks = list(unique_chunks_map.values())

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]

        # Chroma metadata must be flat primitives (str, int, float, bool)
        metadatas = [
            {
                "source_path": c.metadata.source_path,
                "doc_type": c.metadata.doc_type.value,
                "page_number": c.metadata.page_number or -1,
                "line_start": c.metadata.line_start or -1,
                "line_end": c.metadata.line_end or -1,
                "section_title": c.metadata.section_title or "",
            }
            for c in chunks
        ]

        print(f"Generating Gemini embeddings for {len(chunks)} chunks...")
        embeddings = self.embedder.embed_documents(documents)

        print(f"Storing {len(chunks)} vectors in ChromaDB...")
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 4) -> list[tuple[DocumentChunk, float]]:
        """Semantic similarity search returning matching DocumentChunks and cosine similarity scores."""
        query_vector = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )

        matched_chunks: list[tuple[DocumentChunk, float]] = []
        if not results["ids"] or not results["ids"][0]:
            return matched_chunks

        for idx in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][idx]
            content = results["documents"][0][idx]
            meta = results["metadatas"][0][idx]
            distance = results["distances"][0][idx] if results.get("distances") else 0.0

            # Cosine similarity = 1 - cosine distance
            similarity = 1.0 - distance

            metadata = DocumentMetadata(
                source_path=meta["source_path"],
                doc_type=DocumentType(meta["doc_type"]),
                page_number=meta["page_number"] if meta["page_number"] != -1 else None,
                line_start=meta["line_start"] if meta["line_start"] != -1 else None,
                line_end=meta["line_end"] if meta["line_end"] != -1 else None,
                section_title=meta["section_title"] if meta["section_title"] else None,
            )

            chunk = DocumentChunk(chunk_id=chunk_id, content=content, metadata=metadata)
            matched_chunks.append((chunk, similarity))

        return matched_chunks

    def count(self) -> int:
        return self.collection.count()
