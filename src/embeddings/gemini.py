from litellm import embedding
from src.core.config import settings


class GeminiEmbedder:
    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        api_key: str | None = settings.GEMINI_API_KEY,
        batch_size: int = 20,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding vector for a single search query."""
        response = embedding(
            model=self.model_name,
            input=[text],
            api_key=self.api_key,
        )
        return response.data[0]["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of document chunks with batching."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = embedding(
                model=self.model_name,
                input=batch,
                api_key=self.api_key,
            )
            # Collect vectors in original order
            batch_embeddings = [item["embedding"] for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
