"""Deterministic and optional integration tests for the Chroma retriever."""

import os
import tempfile
import unittest
import uuid
from pathlib import Path

import chromadb
from langchain_ollama import OllamaEmbeddings

from agents.business_agent.retriever import (
    ChromaKnowledgeRetriever,
    KnowledgeChunk,
    KnowledgeIndexEmptyError,
    load_knowledge_chunks,
)


class FakeEmbedder:
    """Map each test document to one of three deterministic topic vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.lower()
        return [
            float("star" in normalized or "behavioral" in normalized),
            float("resume" in normalized),
            float("network" in normalized),
        ]


class BusinessRetrieverTests(unittest.TestCase):
    def create_retriever(self) -> ChromaKnowledgeRetriever:
        return ChromaKnowledgeRetriever(
            client=chromadb.EphemeralClient(),
            collection_name=f"business_retriever_{uuid.uuid4().hex}",
            embedder=FakeEmbedder(),
            batch_size=2,
        )

    @staticmethod
    def sample_chunks() -> list[KnowledgeChunk]:
        return [
            KnowledgeChunk(
                title="Interview Preparation",
                source="interview_preparation.md",
                content="Use STAR for behavioral interview answers.",
            ),
            KnowledgeChunk(
                title="Resume Guidance",
                source="resume_guidance.md",
                content="Tailor each resume bullet to the role.",
            ),
            KnowledgeChunk(
                title="Job Search Workflow",
                source="job_search_workflow.md",
                content="Build a network and track applications.",
            ),
        ]

    def test_load_knowledge_chunks_reads_markdown_and_relative_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "interview_notes.md").write_text(
                "# Interview Notes\n\n"
                "## Behavioral\n\n"
                "Use STAR to structure an answer.",
                encoding="utf-8",
            )

            chunks = load_knowledge_chunks(root)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Interview Notes")
        self.assertEqual(chunks[0].source, "nested/interview_notes.md")
        self.assertIn("Use STAR", chunks[0].content)

    def test_upsert_is_idempotent_and_preserves_source_metadata(self):
        retriever = self.create_retriever()
        chunks = self.sample_chunks()

        self.assertEqual(retriever.index_documents(chunks), 3)
        collection = retriever._get_collection()
        first_records = collection.get(include=["metadatas"])
        self.assertEqual(retriever.index_documents(chunks), 3)
        second_records = collection.get(include=["metadatas"])

        self.assertEqual(set(first_records["ids"]), set(second_records["ids"]))
        metadata_by_id = dict(zip(second_records["ids"], second_records["metadatas"]))
        interview_metadata = metadata_by_id[
            next(
                record_id
                for record_id, metadata in metadata_by_id.items()
                if metadata["source"] == "interview_preparation.md"
            )
        ]
        self.assertEqual(interview_metadata["title"], "Interview Preparation")
        self.assertEqual(interview_metadata["chunk_index"], 0)

    def test_retrieve_returns_nearest_chunk_with_cosine_score(self):
        retriever = self.create_retriever()
        retriever.index_documents(self.sample_chunks())

        results = retriever.retrieve("STAR behavioral interview", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "interview_preparation.md")
        self.assertAlmostEqual(results[0].score, 1.0, places=5)

    def test_empty_collection_raises_actionable_diagnostic(self):
        retriever = self.create_retriever()

        with self.assertRaisesRegex(
            KnowledgeIndexEmptyError,
            "scripts/ingest_knowledge.py",
        ):
            retriever.retrieve("How should I prepare for an interview?")

    def test_rebuild_replaces_existing_collection(self):
        retriever = self.create_retriever()
        retriever.index_documents(self.sample_chunks()[:1])
        replacement = KnowledgeChunk(
            title="Updated Resume Guidance",
            source="resume_guidance.md",
            content="Use evidence-backed resume achievements.",
        )

        count = retriever.index_documents([replacement], rebuild=True)
        records = retriever._get_collection().get(include=["documents"])

        self.assertEqual(count, 1)
        self.assertEqual(records["documents"], [replacement.content])


@unittest.skipUnless(
    os.getenv("RUN_RAG_INTEGRATION") == "1",
    "Set RUN_RAG_INTEGRATION=1 to call real Ollama embeddings.",
)
class ChromaOllamaIntegrationTests(unittest.TestCase):
    def test_bge_m3_index_survives_a_new_persistent_client(self):
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        collection_name = f"career_knowledge_{uuid.uuid4().hex}"
        chunks = load_knowledge_chunks()
        self.assertTrue(chunks)

        with tempfile.TemporaryDirectory() as directory:
            retriever = ChromaKnowledgeRetriever(
                directory=directory,
                collection_name=collection_name,
                embedding_model=embedding_model,
                base_url=base_url,
            )
            first_client = retriever._get_client()
            second_client = None
            try:
                indexed_count = retriever.index_documents(chunks, rebuild=True)
                self.assertEqual(indexed_count, len(chunks))
                self.assertEqual(retriever.index_documents(chunks), indexed_count)

                second_client = chromadb.PersistentClient(path=directory)
                second_retriever = ChromaKnowledgeRetriever(
                    client=second_client,
                    collection_name=collection_name,
                    embedder=OllamaEmbeddings(
                        model=embedding_model,
                        base_url=base_url,
                    ),
                )
                results = second_retriever.retrieve(
                    "STAR method for behavioral interview answers",
                    limit=2,
                )

                self.assertEqual(second_retriever.count(), len(chunks))
                self.assertTrue(results)
                self.assertEqual(
                    results[0].source,
                    "interview_preparation.md",
                )
            finally:
                if second_client is not None:
                    second_client.close()
                first_client.close()


if __name__ == "__main__":
    unittest.main()
