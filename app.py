import time
from pathlib import Path
import streamlit as st

from src.core.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.vectorstore.chroma import ChromaVectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import RAGGenerator

# --- Page Configuration ---
st.set_page_config(
    page_title=f"{settings.PROJECT_NAME} | Enterprise Neural RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Enterprise CSS Design System ---
st.markdown(
    """
    <style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Clean subtle borders and dark panels */
    .stApp {
        background-color: #0b0f17;
        color: #e2e8f0;
    }

    /* Header Bar */
    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 0 1rem 0;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 1.5rem;
    }

    .brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .brand-tag {
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        background: #1e293b;
        color: #94a3b8;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #334155;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        color: #10b981;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 4px 10px;
        border-radius: 9999px;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }

    .sidebar-section-header {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin: 1.25rem 0 0.5rem 0;
    }

    .topology-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 0.8rem;
    }

    .topology-label {
        color: #94a3b8;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .topology-value {
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        margin-top: 2px;
    }

    /* Citation Card */
    .citation-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 3px solid #3b82f6;
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 8px;
        font-size: 0.82rem;
    }

    .citation-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #93c5fd;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
    }

    /* Chat Elements */
    .stChatMessage {
        background-color: transparent !important;
        border-bottom: 1px solid #1e293b;
        padding: 1.25rem 0 !important;
    }

    .latency-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #64748b;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def init_rag_kernel():
    """Initializes and caches core RAG singletons."""
    pipeline = IngestionPipeline()
    vector_store = ChromaVectorStore()
    generator = RAGGenerator()

    # Ingest baseline knowledge if empty
    if vector_store.count() == 0 and Path("todo.md").exists():
        chunks = pipeline.ingest_file(Path("todo.md"))
        vector_store.upsert_chunks(chunks)

    all_chunks = vector_store.get_all_chunks()
    bm25 = BM25Retriever(all_chunks)
    hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25)

    return pipeline, vector_store, generator, bm25, hybrid_retriever


pipeline, vector_store, generator, bm25, hybrid_retriever = init_rag_kernel()

# --- Sidebar: Operational Topology & Ingestion ---
with st.sidebar:
    st.markdown('<div class="brand-title">NexusDocs</div>', unsafe_allow_html=True)
    st.caption("Enterprise Context Retrieval & Grounding Kernel")

    st.markdown('<div class="sidebar-section-header">System Topology</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="topology-card">
            <div class="topology-label">Primary Inference Model</div>
            <div class="topology-value">{settings.LLM_MODEL.split('/')[-1]}</div>
        </div>
        <div class="topology-card">
            <div class="topology-label">Dense Vector Embeddings</div>
            <div class="topology-value">{settings.EMBEDDING_MODEL.split('/')[-1]} ({settings.EMBEDDING_DIMENSION}d)</div>
        </div>
        <div class="topology-card">
            <div class="topology-label">Retrieval Engine</div>
            <div class="topology-value">Hybrid (Dense Cosine + BM25 RRF)</div>
        </div>
        <div class="topology-card">
            <div class="topology-label">Indexed Corpus Vectors</div>
            <div class="topology-value">{vector_store.count()} chunks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-header">Corpus Ingestion Control</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload source files to vector store",
        type=["pdf", "md", "txt", "py", "js", "ts", "go", "java"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Ingest Corpus Files", use_container_width=True, type="primary"):
            with st.status("Ingesting document corpus...", expanded=False) as status:
                upload_dir = Path("./data/uploads")
                upload_dir.mkdir(parents=True, exist_ok=True)
                new_chunks_count = 0

                for uploaded_file in uploaded_files:
                    file_path = upload_dir / uploaded_file.name
                    file_path.write_bytes(uploaded_file.getbuffer())

                    chunks = pipeline.ingest_file(file_path)
                    if chunks:
                        vector_store.upsert_chunks(chunks)
                        new_chunks_count += len(chunks)

                # Synchronize BM25 sparse index
                all_chunks = vector_store.get_all_chunks()
                bm25.index_chunks(all_chunks)
                status.update(label=f"Ingested {new_chunks_count} chunks into corpus.", state="complete")

            st.rerun()

    # Active Documents List
    all_chunks = vector_store.get_all_chunks()
    unique_sources = sorted(list({c.metadata.source_path for c in all_chunks}))
    if unique_sources:
        st.markdown('<div class="sidebar-section-header">Indexed Documents</div>', unsafe_allow_html=True)
        with st.expander(f"Corpus Catalog ({len(unique_sources)} files)", expanded=False):
            for s in unique_sources:
                st.code(Path(s).name, language="text")


# --- Main Application Header ---
st.markdown(
    """
    <div class="brand-header">
        <div class="brand-title">
            NexusDocs Knowledge Engine
            <span class="brand-tag">v0.1.0-enterprise</span>
        </div>
        <div class="status-pill">
            <div class="status-dot"></div>
            SYS_ONLINE · HYBRID_RRF
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Session State for Conversation Management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "NexusDocs system initialized. All queries are evaluated strictly against verified corpus documentation with deterministic citations.",
            "sources": [],
            "latency_ms": None,
        }
    ]

# Render Message Trajectory
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display Sources Panel
        if msg.get("sources"):
            with st.expander(f"Retrieved Evidence [{len(msg['sources'])} chunks]", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    meta = src.metadata
                    source_str = f"SOURCE: {Path(meta.source_path).name}"
                    if meta.page_number:
                        source_str += f" | PAGE: {meta.page_number}"
                    if meta.line_start and meta.line_end:
                        source_str += f" | LINES: {meta.line_start}-{meta.line_end}"

                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span>[REFERENCE #{idx}] {source_str}</span>
                                <span>TYPE: {meta.doc_type.value.upper()}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.code(src.content, language=meta.doc_type.value if meta.doc_type.value != "pdf" else "text")

        if msg.get("latency_ms"):
            st.markdown(
                f'<div class="latency-badge">Execution latency: {msg["latency_ms"]} ms · Hybrid RRF retrieval · Zero-hallucination guard</div>',
                unsafe_allow_html=True,
            )

# Query Input Loop
if user_prompt := st.chat_input("Enter technical query or search requirement..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        start_time = time.time()

        # Hybrid dense + sparse retrieval
        retrieved_chunks = hybrid_retriever.retrieve(user_prompt, top_k=3)

        # Stream response
        token_stream = generator.stream_generate(user_prompt, retrieved_chunks)
        full_text = st.write_stream(token_stream)

        latency_ms = round((time.time() - start_time) * 1000, 1)

        # Citations drawer
        if retrieved_chunks:
            with st.expander(f"Retrieved Evidence [{len(retrieved_chunks)} chunks]", expanded=False):
                for idx, src in enumerate(retrieved_chunks, 1):
                    meta = src.metadata
                    source_str = f"SOURCE: {Path(meta.source_path).name}"
                    if meta.page_number:
                        source_str += f" | PAGE: {meta.page_number}"
                    if meta.line_start and meta.line_end:
                        source_str += f" | LINES: {meta.line_start}-{meta.line_end}"

                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span>[REFERENCE #{idx}] {source_str}</span>
                                <span>TYPE: {meta.doc_type.value.upper()}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.code(src.content, language=meta.doc_type.value if meta.doc_type.value != "pdf" else "text")

        st.markdown(
            f'<div class="latency-badge">Execution latency: {latency_ms} ms · Hybrid RRF retrieval · Zero-hallucination guard</div>',
            unsafe_allow_html=True,
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_text,
            "sources": retrieved_chunks,
            "latency_ms": latency_ms,
        }
    )
