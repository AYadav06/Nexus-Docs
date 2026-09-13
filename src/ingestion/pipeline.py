from pathlib import Path
from src.core.models import DocumentChunk
from src.ingestion.loaders import PDFLoader, MarkdownLoader, CodeLoader, BaseLoader


class IngestionPipeline:
    def __init__(self):
        self.loaders: dict[str, BaseLoader] = {
            ".pdf": PDFLoader(),
            ".md": MarkdownLoader(),
            ".txt": MarkdownLoader(),
            # Code extensions
            ".py": CodeLoader(),
            ".js": CodeLoader(),
            ".ts": CodeLoader(),
            ".go": CodeLoader(),
            ".rs": CodeLoader(),
            ".java": CodeLoader(),
            ".cpp": CodeLoader(),
        }

    def get_loader_for_file(self, file_path: Path) -> BaseLoader | None:
        return self.loaders.get(file_path.suffix.lower())

    def ingest_file(self, file_path: Path) -> list[DocumentChunk]:
        loader = self.get_loader_for_file(file_path)
        if not loader:
            print(f"Skipping unsupported file: {file_path.name}")
            return []
        return loader.load(file_path)

    def ingest_directory(self, dir_path: Path) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []
        supported_extensions = set(self.loaders.keys())

        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                chunks = self.ingest_file(file_path)
                all_chunks.extend(chunks)

        return all_chunks
