import sys
from pathlib import Path
from src.core.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.vectorstore.chroma import ChromaVectorStore
from src.generation.generator import RAGGenerator


def main():
    print(f"Initializing {settings.PROJECT_NAME}...")
    pipeline = IngestionPipeline()
    vector_store = ChromaVectorStore()
    generator = RAGGenerator()

    # If DB is empty, auto-ingest todo.md as sample knowledge
    if vector_store.count() == 0:
        print("Knowledge base is empty. Auto-indexing todo.md...")
        chunks = pipeline.ingest_file(Path("todo.md"))
        vector_store.upsert_chunks(chunks)

    print(f"Ready! ({vector_store.count()} chunks in knowledge base)")
    print("Ask questions about your docs (type 'exit' or 'q' to quit):\n" + "=" * 60)

    while True:
        try:
            query = input("\n You: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "q", "quit"):
                print("Goodbye!")
                break

            # 1. Retrieve top 3 relevant chunks
            print(" Searching knowledge base...")
            results = vector_store.search(query, top_k=3)
            chunks = [chunk for chunk, _ in results]

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
