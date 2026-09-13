# Production-Grade Multi-Source RAG System — Implementation Roadmap

A step-by-step roadmap to build an industry-grade Retrieval-Augmented Generation (RAG) Q&A system over multi-format documents (PDFs, codebases, markdown/company wikis) with hybrid search, reranking, citations, evaluation, and observability.

---

## 🏗️ System Architecture

```text
[ Document Sources ]
  ├── PDFs (PyMuPDF / pdfplumber)
  ├── Codebases (Tree-sitter / Code-Aware AST Splitters)
  └── Company Docs (Markdown / Structured Docs)
             │
             ▼
[ Ingestion & Processing Pipeline ]
  ├── Text Extraction & Table Parsing
  ├── Structure-Aware Chunking (Contextual / Header-Aware)
  └── Rich Metadata Enrichment (source, page, file_type, commit/date)
             │
             ▼
[ Embedding & Storage Layer ]
  ├── Dense Embeddings (e.g., text-embedding-3 / BGE / nomic-embed)
  ├── Sparse Inverted Index (BM25 for exact keyword/code tokens)
  └── Vector DB (Chroma / Qdrant / LanceDB)
             │
             ▼
[ Query & Retrieval Engine ] ◄── User Query
  ├── Query Rewrite / HyDE (Hypothetical Document Embeddings - optional)
  ├── Hybrid Retrieval (Dense Vector Search + BM25 Sparse Search)
  ├── Reciprocal Rank Fusion (RRF)
  └── Cross-Encoder Reranker (e.g., BGE-Reranker / Cohere)
             │
             ▼
[ Generation & Grounding ]
  ├── Context Assembly & Token Budget Management
  ├── Grounded Prompt with Strict Citation Rules [Source, Page/Line]
  └── LLM (Gemini / Claude / OpenAI / Local Ollama) ──► Streaming Answer
             │
             ▼
[ Evaluation & Observability ]
  ├── RAG Triad: Context Relevance, Groundedness (Faithfulness), Answer Relevance
  └── Tracing & Latency Monitoring (Langfuse / OpenLLMetry)
```

---

## 📋 Phase-by-Phase Checklist

### Phase 1: Project Setup & Architecture Foundations
- [x] **1.1 Virtual Environment & Dependency Management**
  - Use `uv` to manage Python 3.13 dependencies.
  - Setup core dependencies (`fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `python-dotenv`).
  - Configure `.env.example` and environment configuration manager.
- [x] **1.2 Project Layout & Modular Folder Structure**
  - `src/core/` (config, logging, exceptions)
  - `src/ingestion/` (document loaders, preprocessors, chunkers)
  - `src/embeddings/` (embedding model wrapper interfaces)
  - `src/vectorstore/` (vector DB interface & implementation)
  - `src/retrieval/` (hybrid retriever, BM25, reranker)
  - `src/generation/` (LLM client, prompt templates, citation generator)
  - `src/api/` (FastAPI routes, schemas, streaming endpoints)
  - `eval/` (test datasets, RAGAS/custom evaluation scripts)
  - `data/` (sample PDFs, code repositories, docs)
- [x] **1.3 Decision Log**: Select Vector DB & Embedding Models
  - Document choice: Local (Chroma/Qdrant/LanceDB) vs Managed.
  - Document choice: Open-source embedding vs API-based.

---

### Phase 2: Multi-Source Document Ingestion & Smart Chunking
- [x] **2.1 Document Loaders**
  - **PDF Loader**: Extract text, detect headers, preserve page numbers; handle tables cleanly (e.g. `pymupdf` or `pdfplumber`).
  - **Code Loader**: Extract code files, detect language (Python, TypeScript, Go), preserve filepath and line number ranges.
  - **Markdown/Company Docs Loader**: Parse headers (`#`, `##`, `###`), frontmatter metadata.
- [x] **2.2 Advanced Chunking Strategies**
  - **Markdown/Doc Chunker**: Header-aware recursive splitting to keep related sections together.
  - **Code-Aware Chunker**: Function/class-boundary splitting (avoids breaking functions in half).
  - **Sliding Window Chunking with Overlap**: Configure token-based overlap (e.g., 512 chunk size, 64 token overlap) to prevent lost context at chunk boundaries.
- [x] **2.3 Metadata Enrichment**
  - Attach metadata to every chunk: `source_path`, `document_type`, `page_number`, `header_path`, `created_at`, `chunk_id`.

---

### Phase 3: Embedding Pipeline & Vector Store Setup
- [x] **3.1 Unified Embedding Interface**
  - Abstract base class `BaseEmbedder` with methods `embed_documents(texts)` and `embed_query(text)`.
  - Implement at least one local provider (e.g. `FastEmbed` / `HuggingFace`) and one API provider (e.g. OpenAI / Gemini / Ollama).
- [x] **3.2 Vector Database Integration**
  - Initialize vector store collection with schema validation and distance metric (Cosine or Inner Product).
  - Implement batch upsert with chunk deduplication (based on content hash).
  - Implement metadata filtering (e.g. filter by `document_type == 'code'` or `source_path == 'quarterly_report.pdf'`).

---

### Phase 4: Production-Grade Retrieval & Reranking (The Hiring Differentiator)
- [x] **4.1 Dense Vector Retrieval**
  - Top-$K$ semantic similarity search returning chunks with relevance scores.
- [x] **4.2 Sparse Keyword Search (BM25)**
  - Build or integrate a BM25 index over the chunks for exact token matching (crucial for error codes, function names, acronyms).
- [x] **4.3 Hybrid Search & Reciprocal Rank Fusion (RRF)**
  - Combine Dense and Sparse retrieval scores using RRF:
    $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  - Retrieve candidate pool (e.g., top 25 candidates).
- [x] **4.4 Cross-Encoder Reranker**
  - Pass the top candidates to a cross-encoder model (e.g., `BAAI/bge-reranker-base` or FlashRank).
  - Trim to final top-$N$ most relevant chunks (e.g., top 3–5) for the context window.

---

### Phase 5: Generation, Prompt Engineering & Strict Citations
- [x] **5.1 Context Assembly & Token Budgeting**
  - Deduplicate overlapping chunks and assemble context within model token budget.
- [x] **5.2 Grounded Prompt Architecture**
  - Craft system prompt with strict rules:
    - Never hallucinate facts outside the provided context.
    - If the context doesn't contain the answer, explicitly state: *"I cannot find this information in the provided documents."*
    - Mandate bracketed citations: `[Doc: <filename>, Page: <page>, Section: <header>]`.
- [x] **5.3 Structured & Streaming Output**
  - Implement token streaming for low Time-to-First-Token (TTFT).
  - Return structured metadata alongside the answer (list of cited sources with confidence scores).

---

### Phase 6: API Layer & User Interface
- [x] **6.1 FastAPI Backend**
  - `POST /api/v1/ingest`: Upload file(s) or folder paths to trigger ingestion.
  - `POST /api/v1/query`: Standard Q&A endpoint.
  - `POST /api/v1/query/stream`: Server-Sent Events (SSE) streaming endpoint.
  - `GET /api/v1/documents`: List indexed documents and metadata.
  - `DELETE /api/v1/documents/{id}`: Remove document and its vectors.
- [x] **6.2 Interactive Web UI**
  - Clean, responsive chat interface with document upload tray, source citation preview drawer, and latency/model metrics.

---

### Phase 7: RAG Evaluation & Benchmarking (Gold Standard for Resumes)
- [x] **7.1 Synthetic & Golden QA Test Dataset**
  - Create a dataset of 15–25 realistic questions, ground-truth context, and expected answers across PDFs and code.
- [x] **7.2 Evaluation Metrics Implementation**
  - **Context Relevance**: Are the retrieved chunks actually relevant to the question?
  - **Faithfulness / Groundedness**: Is the answer derived *only* from the context?
  - **Answer Relevance**: Does the generated answer directly address the question asked?
- [x] **7.3 Automated Eval Runner**
  - Script to run queries against test set and compute metrics score table.

---

### Phase 8: Observability, Edge Cases & Polish
- [x] **8.1 Tracing & Logging**
  - Instrument retrieval latency, token usage, and search hit distribution.
- [x] **8.2 Edge Case Handling**
  - Empty search results, corrupt PDFs, queries outside the domain, oversized documents.
- [x] **8.3 Documentation & Demo**
  - Architecture diagram, benchmarking results, quickstart guide, and video/GIF demo for GitHub portfolio.

---

## 💡 Tech Stack Recommendations & Trade-offs

| Component | Recommended Choice | Alternative | Why It Matters in Interviews |
| :--- | :--- | :--- | :--- |
| **Vector DB** | **Qdrant** or **Chroma** | LanceDB, FAISS | Fast local development, payload/metadata filtering, production scale |
| **Embeddings** | **BGE-Large** / **text-embedding-3-small** | nomic-embed-text | Understanding dimension size, normalization, cosine vs dot product |
| **Reranker** | **FlashRank** / **BGE-Reranker** | Cohere Rerank API | Demonstrates you know naive vector search alone misses exact keywords |
| **Hybrid Search**| **BM25 + Dense (RRF)** | Dense-only | Solves the #1 industry complaint: vector search failing on exact IDs / code symbols |
| **Parser** | **PyMuPDF (fitz)** / **Tree-sitter** | LangChain standard loaders | Custom loaders show you understand ASTs, document layouts, and chunk boundaries |
| **API** | **FastAPI** | Flask, Express | Standard for high-performance AI microservices with async SSE streaming |
| **Evaluation** | **RAGAS** / Custom LLM-as-Judge | Manual review | Demonstrates quantitative evaluation mindset vs "vibes-based" AI |
