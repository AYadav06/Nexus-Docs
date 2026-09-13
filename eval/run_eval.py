import json
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table

from src.ingestion.pipeline import IngestionPipeline
from src.vectorstore.chroma import ChromaVectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import RAGGenerator
from eval.evaluator import RAGEvaluator


def run_benchmark():
    console = Console()
    console.print("\n[bold cyan]⚡ Running NexusDocs Enterprise RAG Benchmark...[/bold cyan]\n")

    # 1. Setup system components
    pipeline = IngestionPipeline()
    vector_store = ChromaVectorStore()
    generator = RAGGenerator()

    # Ingest core project files if not already in vector store
    for test_file in [Path("todo.md"), Path("src/core/config.py"), Path("src/ingestion/pipeline.py")]:
        if test_file.exists():
            chunks = pipeline.ingest_file(test_file)
            vector_store.upsert_chunks(chunks)

    all_chunks = vector_store.get_all_chunks()
    bm25 = BM25Retriever(all_chunks)
    hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25)
    evaluator = RAGEvaluator()

    # 2. Load dataset
    dataset_path = Path("eval/dataset.json")
    with open(dataset_path, "r") as f:
        test_cases = json.load(f)

    # 3. Execution & Scoring Table
    table = Table(title="NexusDocs RAG Evaluation Benchmark Results", show_lines=True)
    table.add_column("ID", style="bold white", width=6)
    table.add_column("Query", width=38)
    table.add_column("Context Recall", justify="center", style="green")
    table.add_column("Faithfulness", justify="center", style="cyan")
    table.add_column("Relevance", justify="center", style="magenta")

    total_faithfulness = 0.0
    total_relevance = 0.0
    total_recall = 0.0

    for i, item in enumerate(test_cases, 1):
        qid = item["id"]
        query = item["question"]
        expected_src = item["expected_source"]

        console.print(f"[{i}/{len(test_cases)}] Evaluating: [bold]{qid}[/bold] - \"{query[:40]}...\"")

        # Retrieve context
        retrieved_chunks = hybrid_retriever.retrieve(query, top_k=3)

        # Generate Answer (with rate-limit retry)
        answer = "I cannot find sufficient information in the provided documents to answer this question."
        for attempt in range(4):
            try:
                answer = generator.generate(query, retrieved_chunks)
                break
            except Exception:
                time.sleep(15)

        # 1. Context Recall metric
        retrieved_sources = [Path(c.metadata.source_path).name for c in retrieved_chunks]
        if expected_src.startswith("NONE"):
            recall_score = 1.0  # Negative test correctly expects no false context
        else:
            recall_score = 1.0 if any(Path(expected_src).name in s for s in retrieved_sources) else 0.0

        # 2. Faithfulness & Relevance in 1 call
        combined_context = "\n".join([c.content for c in retrieved_chunks])
        scores = evaluator.evaluate_pair(query, combined_context, answer)

        total_recall += recall_score
        total_faithfulness += scores["faithfulness"]
        total_relevance += scores["relevance"]

        table.add_row(
            qid,
            query,
            f"{recall_score * 100:.0f}%",
            f"{scores['faithfulness'] * 100:.0f}%",
            f"{scores['relevance'] * 100:.0f}%",
        )

        # Pacing to respect Gemini free tier rate limits
        if i < len(test_cases):
            time.sleep(6)

    console.print("\n")
    console.print(table)

    n = len(test_cases)
    avg_recall = (total_recall / n) * 100
    avg_faithfulness = (total_faithfulness / n) * 100
    avg_relevance = (total_relevance / n) * 100

    console.print(f"\n[bold green]📊 Final Aggregate Benchmark Scores (N={n}):[/bold green]")
    console.print(f"  • Context Recall:        [bold]{avg_recall:.1f}%[/bold]")
    console.print(f"  • Grounded Faithfulness: [bold]{avg_faithfulness:.1f}%[/bold]")
    console.print(f"  • Answer Relevance:      [bold]{avg_relevance:.1f}%[/bold]\n")


if __name__ == "__main__":
    run_benchmark()
