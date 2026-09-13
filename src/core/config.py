from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "NexusDocs"
    DEBUG: bool = False

    # Generation (LLM)
    LLM_MODEL: str = "gemini/gemini-2.5-flash"
    OPENROUTER_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    LLM_TEMPERATURE: float = 0.1

    # Embeddings (Gemini)
    EMBEDDING_MODEL: str = "gemini/gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 3072

    # Vector Store Settings
    VECTOR_DB_DIR: Path = Path("./data/vector_db")

    # Ingestion / Chunking defaults
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
