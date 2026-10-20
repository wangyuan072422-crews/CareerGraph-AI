"""Run the shared application graph with the configured MySQL checkpointer.

This example reads database credentials from the project environment through
``utils.checkpointer``. It never embeds a MySQL user or password.
"""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage

from graphs.application import build_application_graph
from utils.checkpointer import close_checkpointer, get_checkpointer
from utils.env import load_environment


def run_with_mysql() -> str:
    """Run one two-turn conversation and return its checkpoint thread ID."""

    load_environment()
    checkpointer = get_checkpointer("mysql")
    graph = build_application_graph(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        first = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "How should I prepare for a behavioral "
                            "interview?"
                        )
                    )
                ],
                "forced_agent": "business_agent",
                "next": None,
                "status": "ok",
                "error": None,
            },
            config=config,
        )
        print(f"First response: {first['messages'][-1].content[:160]}")

        second = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Can you give me a concrete STAR example?"
                    )
                ],
                "forced_agent": "business_agent",
                "next": None,
                "status": "ok",
                "error": None,
            },
            config=config,
        )
        print(f"Follow-up response: {second['messages'][-1].content[:160]}")

        state = graph.get_state(config)
        message_count = len(state.values.get("messages", []))
        print(f"Thread {thread_id} contains {message_count} saved messages.")
        return thread_id
    finally:
        close_checkpointer(checkpointer)


if __name__ == "__main__":
    print("Using CHECKPOINTER_BACKEND=mysql through get_checkpointer('mysql').")
    run_with_mysql()
