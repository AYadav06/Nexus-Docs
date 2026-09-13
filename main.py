import sys
from pathlib import Path
from src.core.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.vectorstore.chroma import ChromaVectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import RAGGenerator


def main():
    print(f" Initializing {settings.PROJECT_NAME} with Hybrid Search (Dense + BM25)...")
    pipeline = IngestionPipeline()
    vector_store = ChromaVectorStore()
    generator = RAGGenerator()

    # 1. If DB is empty, auto-ingest todo.md as sample knowledge
    if vector_store.count() == 0:
        print(" Knowledge base is empty. Auto-indexing todo.md...")
        chunks = pipeline.ingest_file(Path("todo.md"))
        vector_store.upsert_chunks(chunks)

    # 2. Reconstruct BM25 Sparse Index from stored Chroma chunks
    all_chunks = vector_store.get_all_chunks()
    bm25 = BM25Retriever(all_chunks)
    hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25)

    print(f" Ready! ({len(all_chunks)} chunks indexed in Vector DB + BM25)")
    print(" Ask questions (supports exact keywords, code symbols, and conceptual queries):\n" + "=" * 60)

    while True:
        try:
            query = input("\n You: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "q", "quit"):
                print("Goodbye!")
                break

            # 1. Hybrid Retrieval (combines Dense Cosine + Sparse BM25 via RRF)
            print(" Running Hybrid Search (Dense + BM25 with RRF)...")
            chunks = hybrid_retriever.retrieve(query, top_k=3)

            # 2. Stream generation
            print("\n NexusDocs: ", end="", flush=True)
            for token in generator.stream_generate(query, chunks):
                print(token, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()

