"""业务 Agent 知识检索的确定性测试。"""
import tempfile
import unittest
import uuid
from pathlib import Path

import chromadb
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agents.business_agent import (
    ChromaKnowledgeRetriever,
    KnowledgeChunk,
    KnowledgeIndexEmptyError,
    create_business_agent_node,
    load_knowledge_chunks,
)


class RecordingChatModel:
    def __init__(self, content: str):
        self.content = content
        self.received_messages: list[BaseMessage] = []

    def invoke(self, messages):
        self.received_messages = list(messages)
        return AIMessage(content=self.content)


class RecordingRetriever:
    def __init__(self, chunks: list[KnowledgeChunk]):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, query: str, *, limit: int = 3):
        self.calls.append((query, limit))
        return self.chunks[:limit]


class BrokenRetriever:
    def retrieve(self, _query: str, *, limit: int = 3):
        raise RuntimeError(f"private retrieval failure with limit {limit}")


class SemanticTestEmbedder:
    """Map related concepts to the same deterministic vector."""

    def _embed(self, text: str) -> list[float]:
        normalized = text.lower()
        if any(term in normalized for term in ("star", "behavioral", "interview answer")):
            return [1.0, 0.0, 0.0]
        if "resume" in normalized:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def create_test_retriever() -> ChromaKnowledgeRetriever:
    return ChromaKnowledgeRetriever(
        client=chromadb.EphemeralClient(),
        collection_name=f"test_business_rag_{uuid.uuid4().hex}",
        embedder=SemanticTestEmbedder(),
    )


class BusinessAgentRagTests(unittest.TestCase):
    def test_node_retrieves_latest_question_and_injects_cited_context(self):
        retriever = RecordingRetriever(
            [
                KnowledgeChunk(
                    title="Behavioral interviews",
                    source="interview_preparation.md",
                    content="Use STAR: Situation, Task, Action, Result.",
                )
            ]
        )
        model = RecordingChatModel("Use STAR and a measurable result.")
        node = create_business_agent_node(llm=model, retriever=retriever)

        result = node(
            {
                "messages": [
                    HumanMessage(content="My previous question"),
                    AIMessage(content="Previous answer"),
                    HumanMessage(content="How should I answer a behavioral question?"),
                ]
            }
        )

        self.assertEqual(
            retriever.calls,
            [("How should I answer a behavioral question?", 3)],
        )
        system_message = model.received_messages[0]
        self.assertEqual(system_message.type, "system")
        self.assertIn("Use STAR", system_message.content)
        self.assertIn("interview_preparation.md", system_message.content)
        answer = result["messages"][0]
        self.assertIn("Use STAR and a measurable result.", answer.content)
        self.assertIn("Sources:", answer.content)
        self.assertEqual(
            answer.additional_kwargs["sources"],
            [{"title": "Behavioral interviews", "source": "interview_preparation.md"}],
        )

    def test_no_results_instructs_the_model_not_to_invent_sources(self):
        retriever = RecordingRetriever([])
        model = RecordingChatModel("I cannot verify that private detail.")
        node = create_business_agent_node(llm=model, retriever=retriever)

        result = node(
            {"messages": [HumanMessage(content="What is the CEO's private calendar?")]}
        )

        system_prompt = model.received_messages[0].content
        self.assertIn("No relevant knowledge-base passage", system_prompt)
        self.assertNotIn("Sources:", result["messages"][0].content)

    def test_retrieval_failure_does_not_leak_internal_error(self):
        model = RecordingChatModel("Answer from conversation only.")
        node = create_business_agent_node(llm=model, retriever=BrokenRetriever())

        with self.assertLogs(
            "agents.business_agent.agent",
            level="ERROR",
        ):
            result = node(
                {"messages": [HumanMessage(content="Prepare me for an interview")]}
            )

        system_prompt = model.received_messages[0].content
        answer = result["messages"][0].content
        self.assertIn("No relevant knowledge-base passage", system_prompt)
        self.assertNotIn("private retrieval failure", system_prompt)
        self.assertNotIn("private retrieval failure", answer)


class ChromaKnowledgeRetrieverTests(unittest.TestCase):
    def test_vector_search_matches_related_concept_without_exact_keyword(self):
        retriever = create_test_retriever()
        retriever.index_documents(
            [
                KnowledgeChunk(
                    title="STAR answers",
                    source="interview_preparation.md",
                    content="STAR method: use Situation, Task, Action, and Result.",
                ),
                KnowledgeChunk(
                    title="Resume evidence",
                    source="resume_guidance.md",
                    content="Tailor resume bullets with measurable evidence.",
                ),
            ]
        )

        results = retriever.retrieve(
            "How do I structure a behavioral interview answer?",
            limit=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "interview_preparation.md")
        self.assertEqual(results[0].title, "STAR answers")
        self.assertIn("Situation", results[0].content)
        self.assertGreater(results[0].score, 0)

    def test_index_documents_is_idempotent(self):
        retriever = create_test_retriever()
        chunks = [
            KnowledgeChunk(
                title="STAR answers",
                source="interview_preparation.md",
                    content="STAR method: use Situation, Task, Action, and Result.",
            ),
            KnowledgeChunk(
                title="Resume evidence",
                source="resume_guidance.md",
                content="Tailor resume bullets with measurable evidence.",
            ),
        ]

        first_count = retriever.index_documents(chunks)
        second_count = retriever.index_documents(chunks)

        self.assertEqual(first_count, 2)
        self.assertEqual(second_count, 2)

    def test_rebuild_removes_obsolete_chunks(self):
        retriever = create_test_retriever()
        retriever.index_documents(
            [
                KnowledgeChunk(
                    title="Old guidance",
                    source="old.md",
                    content="Old resume guidance.",
                )
            ]
        )

        count = retriever.index_documents(
            [
                KnowledgeChunk(
                    title="STAR answers",
                    source="interview_preparation.md",
                    content="STAR method: use Situation, Task, Action, and Result.",
                )
            ],
            rebuild=True,
        )

        results = retriever.retrieve("behavioral interview answer", limit=5)

        self.assertEqual(count, 1)
        self.assertEqual([chunk.source for chunk in results], ["interview_preparation.md"])

    def test_empty_index_raises_clear_error(self):
        retriever = create_test_retriever()

        with self.assertRaisesRegex(KnowledgeIndexEmptyError, "ingest_knowledge"):
            retriever.retrieve("STAR method")

    def test_load_knowledge_chunks_preserves_heading_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interview_preparation.md"
            path.write_text(
                "# Interview Preparation\n\nUse the STAR method.",
                encoding="utf-8",
            )

            chunks = load_knowledge_chunks(directory)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Interview Preparation")
        self.assertEqual(chunks[0].source, "interview_preparation.md")
        self.assertEqual(chunks[0].content, "Use the STAR method.")


if __name__ == "__main__":
    unittest.main()
