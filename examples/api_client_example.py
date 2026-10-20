"""Client examples for the career assistant HTTP API."""

from __future__ import annotations

import json
from typing import Any

import requests


BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 180


def manual_query(
    message: str,
    *,
    agent_type: str = "business",
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Call the caller-selected Agent endpoint."""

    payload: dict[str, Any] = {
        "message": message,
        "agent_type": agent_type,
    }
    if thread_id:
        payload["thread_id"] = thread_id

    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def auto_query(
    message: str,
    *,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Call the Supervisor-routed endpoint."""

    payload: dict[str, Any] = {"message": message}
    if thread_id:
        payload["thread_id"] = thread_id

    response = requests.post(
        f"{BASE_URL}/query/auto",
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def stream_query(
    message: str,
    *,
    agent_type: str = "business",
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Consume SSE events and return the final JSON payload."""

    payload: dict[str, Any] = {
        "message": message,
        "agent_type": agent_type,
    }
    if thread_id:
        payload["thread_id"] = thread_id

    final_payload: dict[str, Any] | None = None
    with requests.post(
        f"{BASE_URL}/query/stream",
        json=payload,
        stream=True,
        timeout=TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()
        event_name = "message"
        for raw_line in response.iter_lines(decode_unicode=True):
            line = raw_line or ""
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
                print(f"SSE event: {event_name}")
                if event_name == "final":
                    final_payload = data
                elif event_name == "error":
                    raise RuntimeError(data["message"])

    if final_payload is None:
        raise RuntimeError("The stream ended without a final event.")
    return final_payload


def get_history(thread_id: str) -> dict[str, Any]:
    """Read checkpoint-backed conversation history."""

    response = requests.get(
        f"{BASE_URL}/conversation/{thread_id}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def delete_history(thread_id: str) -> dict[str, Any]:
    """Delete all checkpoints for one conversation."""

    response = requests.delete(
        f"{BASE_URL}/conversation/{thread_id}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Exercise manual, automatic, and streamed calls in one conversation."""

    health = requests.get(BASE_URL, timeout=10)
    health.raise_for_status()
    print(health.json())

    manual = manual_query(
        "How should I prepare for a behavioral interview?",
        agent_type="business",
    )
    thread_id = manual["thread_id"]
    print("Manual route:", manual["agent_type"], manual["status"])

    automatic = auto_query(
        "Show my saved job applications with limit 10.",
        thread_id=thread_id,
    )
    print("Automatic route:", automatic["agent_type"], automatic["status"])

    final = stream_query(
        "Give me one concise interview preparation tip.",
        agent_type="business",
        thread_id=thread_id,
    )
    print("Stream final:", final["agent_type"], final["status"])

    history = get_history(thread_id)
    print("History message count:", len(history["messages"]))

    print(delete_history(thread_id))


if __name__ == "__main__":
    main()
