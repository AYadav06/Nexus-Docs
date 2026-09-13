from typing import Generator
from litellm import completion
from src.core.config import settings
from src.core.models import DocumentChunk
from src.generation.prompts import RAG_SYSTEM_PROMPT, format_context


class RAGGenerator:
    def __init__(
        self,
        model_name: str = settings.LLM_MODEL,
        api_key: str | None = settings.GEMINI_API_KEY,
        temperature: float = settings.LLM_TEMPERATURE,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature

    def _build_messages(self, query: str, context_chunks: list[DocumentChunk]) -> list[dict]:
        formatted_context = format_context(context_chunks)
        system_instruction = RAG_SYSTEM_PROMPT.format(context=formatted_context)

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": query},
        ]

    def generate(self, query: str, context_chunks: list[DocumentChunk]) -> str:
        """Generates a complete grounded response."""
        messages = self._build_messages(query, context_chunks)

        response = completion(
            model=self.model_name,
            messages=messages,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def stream_generate(self, query: str, context_chunks: list[DocumentChunk]) -> Generator[str, None, None]:
        """Streams tokens in real-time for low Time-to-First-Token (TTFT)."""
        messages = self._build_messages(query, context_chunks)

        response = completion(
            model=self.model_name,
            messages=messages,
            api_key=self.api_key,
            temperature=self.temperature,
            stream=True,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
