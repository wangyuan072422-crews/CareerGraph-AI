"""Local Chroma vector retrieval for the business knowledge base."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chromadb
from chromadb.errors import NotFoundError
from langchain_ollama import OllamaEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "career_knowledge"
DEFAULT_CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma"
DEFAULT_COLLECTION_NAME = "career_knowledge_v1"
DEFAULT_EMBEDDING_MODEL = "bge-m3:latest"
DEFAULT_BATCH_SIZE = 64
MAX_RESULTS = 10


@dataclass(frozen=True)
class KnowledgeChunk:
    """A retrievable passage with metadata suitable for source citations."""

    title: str
    source: str
    content: str
    score: float = 0.0


class KnowledgeRetriever(Protocol):
    """Minimal retriever contract consumed by the business Agent."""

    def retrieve(self, query: str, *, limit: int = 3) -> list[KnowledgeChunk]:
        ...


class KnowledgeIndexEmptyError(RuntimeError):
    """Raised when a query targets an index that has not been populated."""


class Embedder(Protocol):
    """Embedding methods used by the Chroma retriever."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def _markdown_title(content: str, fallback: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", content)
    return match.group(1).strip() if match else fallback


def _split_markdown(
    content: str,
    *,
    title: str,
    source: str,
    max_chars: int = 700,
) -> list[KnowledgeChunk]:
    """Split Markdown paragraphs into bounded passages."""

    body = re.sub(r"(?m)^#\s+.+?$", "", content, count=1).strip()
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", body)
        if paragraph.strip()
    ]

    chunks: list[KnowledgeChunk] = []
    current: list[str] = []
    current_length = 0
    for paragraph in paragraphs:
        if current and current_length + len(paragraph) + 2 > max_chars:
            chunks.append(KnowledgeChunk(title, source, "\n\n".join(current)))
            current = []
            current_length = 0
        current.append(paragraph)
        current_length += len(paragraph) + 2

    if current:
        chunks.append(KnowledgeChunk(title, source, "\n\n".join(current)))
    return chunks


def _resolve_project_path(value: str | Path | None, default: Path) -> Path:
    path = Path(value) if value is not None else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_knowledge_chunks(
    directory: str | Path | None = None,
) -> list[KnowledgeChunk]:
    """Load and chunk all Markdown documents under a knowledge directory."""

    knowledge_directory = _resolve_project_path(directory, DEFAULT_KNOWLEDGE_DIR)
    if not knowledge_directory.is_dir():
        return []

    chunks: list[KnowledgeChunk] = []
    for path in sorted(knowledge_directory.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        source = path.relative_to(knowledge_directory).as_posix()
        title = _markdown_title(content, path.stem.replace("_", " ").title())
        chunks.extend(
            _split_markdown(
                content,
                title=title,
                source=source,
            )
        )
    return chunks


def _stable_chunk_id(source: str, chunk_index: int, content: str) -> str:
    """Build an idempotent Chroma id from the source and passage content."""

    identity = f"{source}\0{chunk_index}\0{content}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def _positive_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


class ChromaKnowledgeRetriever:
    """Persist career documents in Chroma and retrieve them by vector similarity."""

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        client: Any | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedder: Embedder | None = None,
    ):
        self.directory = _resolve_project_path(directory, DEFAULT_CHROMA_DIRECTORY)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.batch_size = max(1, int(batch_size))
        self._client = client
        self._embedder = embedder
        self._collection: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.directory))
        return self._client

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.base_url,
            )
        return self._embedder

    def _get_collection(self) -> Any:
        if self._collection is None:
            self._collection = self._get_client().get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        return self._collection

    def _drop_collection(self) -> None:
        try:
            self._get_client().delete_collection(self.collection_name)
        except NotFoundError:
            pass
        self._collection = None

    def count(self) -> int:
        """Return the number of indexed passages."""

        return int(self._get_collection().count())

    def index_documents(
        self,
        chunks: Iterable[KnowledgeChunk],
        *,
        rebuild: bool = False,
    ) -> int:
        """Upsert passages and return the resulting collection size."""

        items = list(chunks)
        if not items:
            raise KnowledgeIndexEmptyError(
                "No knowledge chunks were found. Add Markdown documents under "
                f"{DEFAULT_KNOWLEDGE_DIR} and run scripts/ingest_knowledge.py."
            )

        if rebuild:
            self._drop_collection()

        collection = self._get_collection()
        source_positions: dict[str, int] = {}
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int]] = []

        for chunk in items:
            chunk_index = source_positions.get(chunk.source, 0)
            source_positions[chunk.source] = chunk_index + 1
            ids.append(_stable_chunk_id(chunk.source, chunk_index, chunk.content))
            documents.append(chunk.content)
            metadatas.append(
                {
                    "title": chunk.title,
                    "source": chunk.source,
                    "chunk_index": chunk_index,
                }
            )

        embedder = self._get_embedder()
        for start in range(0, len(items), self.batch_size):
            end = start + self.batch_size
            batch_documents = documents[start:end]
            embeddings = embedder.embed_documents(batch_documents)
            if len(embeddings) != len(batch_documents):
                raise ValueError("Embedding provider returned an unexpected batch size.")
            collection.upsert(
                ids=ids[start:end],
                documents=batch_documents,
                metadatas=metadatas[start:end],
                embeddings=embeddings,
            )

        return int(collection.count())

    def retrieve(self, query: str, *, limit: int = 3) -> list[KnowledgeChunk]:
        """Embed a query and return the closest indexed passages."""

        normalized_query = query.strip()
        if not normalized_query or limit <= 0:
            return []

        collection = self._get_collection()
        indexed_count = int(collection.count())
        if indexed_count == 0:
            raise KnowledgeIndexEmptyError(
                "The Chroma knowledge index is empty. Run "
                "python scripts/ingest_knowledge.py before querying."
            )

        result_limit = min(limit, MAX_RESULTS, indexed_count)
        query_embedding = self._get_embedder().embed_query(normalized_query)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=result_limit,
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[KnowledgeChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if document is None:
                continue
            values = metadata or {}
            source = str(values.get("source", ""))
            title = str(values.get("title", Path(source).stem))
            score = 1.0 - float(distance)
            chunks.append(
                KnowledgeChunk(
                    title=title,
                    source=source,
                    content=str(document),
                    score=score,
                )
            )

        chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        return chunks[:result_limit]


def create_default_business_retriever() -> ChromaKnowledgeRetriever:
    """Create the configured local Chroma retriever without rebuilding it."""

    return ChromaKnowledgeRetriever(
        directory=os.getenv("CHROMA_PERSIST_DIRECTORY", str(DEFAULT_CHROMA_DIRECTORY)),
        collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME",
            DEFAULT_COLLECTION_NAME,
        ),
        embedding_model=os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        ),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        batch_size=_positive_int(
            os.getenv("CHROMA_BATCH_SIZE"),
            DEFAULT_BATCH_SIZE,
        ),
    )


__all__ = [
    "ChromaKnowledgeRetriever",
    "DEFAULT_CHROMA_DIRECTORY",
    "DEFAULT_COLLECTION_NAME",
    "KnowledgeChunk",
    "KnowledgeIndexEmptyError",
    "KnowledgeRetriever",
    "create_default_business_retriever",
    "load_knowledge_chunks",
]
