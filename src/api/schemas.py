from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve")


class SourceCitation(BaseModel):
    source_path: str
    doc_type: str
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    latency_ms: float


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    total_vectors: int


class StatsResponse(BaseModel):
    total_chunks: int
    unique_documents: list[str]
