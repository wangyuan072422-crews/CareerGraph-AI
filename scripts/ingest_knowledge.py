"""Ingest career Markdown documents into the local Chroma index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.business_agent.retriever import (  # noqa: E402
    KnowledgeIndexEmptyError,
    create_default_business_retriever,
    load_knowledge_chunks,
)
from utils.env import load_environment  # noqa: E402


def main() -> int:
    """Load, embed, and upsert the project knowledge documents."""

    load_environment()
    parser = argparse.ArgumentParser(
        description="Build or update the local career knowledge vector index."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the collection before ingesting, then rebuild it from scratch.",
    )
    args = parser.parse_args()

    chunks = load_knowledge_chunks()
    if not chunks:
        print("No Markdown knowledge chunks were found; the index was not changed.")
        return 1

    retriever = create_default_business_retriever()
    try:
        count = retriever.index_documents(chunks, rebuild=args.rebuild)
    except KnowledgeIndexEmptyError as exc:
        print(str(exc))
        return 1
    except Exception as exc:
        print(
            "Knowledge ingestion failed. Check Ollama bge-m3 and Chroma "
            f"configuration: {type(exc).__name__}: {exc}"
        )
        return 2

    action = "Rebuilt" if args.rebuild else "Indexed"
    print(f"{action} {len(chunks)} knowledge chunks; collection count: {count}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
