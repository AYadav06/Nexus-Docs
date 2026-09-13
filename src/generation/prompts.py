from src.core.models import DocumentChunk

RAG_SYSTEM_PROMPT = """You are NexusDocs, a precise and grounded enterprise documentation assistant.

Your task is to answer the user's question using ONLY the provided CONTEXT blocks below.

STRICT GROUNDING GUIDELINES:
1. Base your answer EXCLUSIVELY on the provided CONTEXT. Do NOT assume, extrapolate, or use outside knowledge.
2. If the CONTEXT does not contain enough information to answer the question, respond with:
   "I cannot find sufficient information in the provided documents to answer this question."
3. Every factual statement must cite its source inline using bracketed format, matching the document source and page/line:
   - For PDFs: `[Doc: <source_path>, Page: <page_number>]`
   - For Code: `[Doc: <source_path>, Lines: <line_start>-<line_end>]`
   - For Docs/Markdown: `[Doc: <source_path>]`
4. Keep the answer clear, technical, well-structured, and concise.

CONTEXT:
{context}
"""


def format_context(chunks: list[DocumentChunk]) -> str:
    """Formats retrieved DocumentChunks into structured context blocks for the LLM."""
    if not chunks:
        return "No relevant documents found."

    blocks: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        source_details = [f"Source: {meta.source_path}"]

        if meta.page_number is not None:
            source_details.append(f"Page: {meta.page_number}")
        if meta.line_start is not None and meta.line_end is not None:
            source_details.append(f"Lines: {meta.line_start}-{meta.line_end}")
        if meta.section_title:
            source_details.append(f"Section: {meta.section_title}")

        header = f"--- Document Block {i} [{', '.join(source_details)}] ---"
        block = f"{header}\n{chunk.content.strip()}"
        blocks.append(block)

    return "\n\n".join(blocks)
