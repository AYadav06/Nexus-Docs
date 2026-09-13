import json
import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.core.config import settings
from src.core.models import DocumentChunk
from src.ingestion.pipeline import IngestionPipeline
from src.vectorstore.chroma import ChromaVectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import RAGGenerator
from src.api.schemas import QueryRequest, QueryResponse, SourceCitation, IngestResponse, StatsResponse

# Global singletons
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize RAG components
    pipeline = IngestionPipeline()
    vector_store = ChromaVectorStore()
    generator = RAGGenerator()

    # Index sample if empty
    if vector_store.count() == 0 and Path("todo.md").exists():
        chunks = pipeline.ingest_file(Path("todo.md"))
        vector_store.upsert_chunks(chunks)

    all_chunks = vector_store.get_all_chunks()
    bm25 = BM25Retriever(all_chunks)
    hybrid_retriever = HybridRetriever(vector_store, bm25)

    state["pipeline"] = pipeline
    state["vector_store"] = vector_store
    state["generator"] = generator
    state["bm25"] = bm25
    state["hybrid_retriever"] = hybrid_retriever

    yield
    state.clear()


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/v1/stats", response_model=StatsResponse)
def get_stats():
    vector_store: ChromaVectorStore = state["vector_store"]
    chunks = vector_store.get_all_chunks()
    sources = sorted(list({c.metadata.source_path for c in chunks}))
    return StatsResponse(total_chunks=len(chunks), unique_documents=sources)


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    pipeline: IngestionPipeline = state["pipeline"]
    vector_store: ChromaVectorStore = state["vector_store"]
    bm25: BM25Retriever = state["bm25"]

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    new_chunks = pipeline.ingest_file(file_path)
    if not new_chunks:
        raise HTTPException(status_code=400, detail="File format unsupported or file empty.")

    # 1. Upsert into ChromaDB
    vector_store.upsert_chunks(new_chunks)

    # 2. Refresh BM25 with all current chunks
    all_chunks = vector_store.get_all_chunks()
    bm25.index_chunks(all_chunks)

    return IngestResponse(
        filename=file.filename,
        chunks_created=len(new_chunks),
        total_vectors=vector_store.count(),
    )


@app.post("/api/v1/query", response_model=QueryResponse)
def query_sync(req: QueryRequest):
    start_time = time.time()
    hybrid_retriever: HybridRetriever = state["hybrid_retriever"]
    generator: RAGGenerator = state["generator"]

    chunks = hybrid_retriever.retrieve(req.query, top_k=req.top_k)
    answer = generator.generate(req.query, chunks)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    citations = [
        SourceCitation(
            source_path=c.metadata.source_path,
            doc_type=c.metadata.doc_type.value,
            page_number=c.metadata.page_number,
            line_start=c.metadata.line_start,
            line_end=c.metadata.line_end,
            snippet=c.content[:250],
        )
        for c in chunks
    ]

    return QueryResponse(answer=answer, sources=citations, latency_ms=latency_ms)


@app.post("/api/v1/query/stream")
def query_stream(req: QueryRequest):
    """Server-Sent Events (SSE) streaming endpoint."""
    hybrid_retriever: HybridRetriever = state["hybrid_retriever"]
    generator: RAGGenerator = state["generator"]

    chunks = hybrid_retriever.retrieve(req.query, top_k=req.top_k)

    def event_stream():
        # First send sources metadata event
        sources_payload = [
            {
                "source_path": c.metadata.source_path,
                "doc_type": c.metadata.doc_type.value,
                "page_number": c.metadata.page_number,
                "line_start": c.metadata.line_start,
                "line_end": c.metadata.line_end,
                "snippet": c.content[:300],
            }
            for c in chunks
        ]
        yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"

        # Next stream token chunks
        for token in generator.stream_generate(req.query, chunks):
            yield f"event: message\ndata: {json.dumps({'token': token})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Static directory for UI
STATIC_DIR = Path("src/api/static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": f"Welcome to {settings.PROJECT_NAME} API. Visit /docs for Swagger."}
