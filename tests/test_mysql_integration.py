"""Optional integration tests for the real MySQL checkpointer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from graphs.application import build_application_graph
from utils.checkpointer import close_checkpointer, get_checkpointer
from utils.env import load_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_test_state(_state: dict) -> dict:
    """Return a deterministic assistant message for the persistence check."""

    return {"messages": [AIMessage(content="mysql-checkpoint-persistence")]}


class MySQLCheckpointerIntegrationTests(unittest.TestCase):
    """Exercise real MySQL persistence without requiring Ollama."""

    @classmethod
    def setUpClass(cls) -> None:
        load_environment()
        if os.getenv("RUN_MYSQL_INTEGRATION") != "1":
            raise unittest.SkipTest(
                "Set RUN_MYSQL_INTEGRATION=1 to use the configured MySQL database."
            )

    def test_write_cross_process_restore_and_delete_thread(self) -> None:
        load_environment()
        checkpointer = None
        thread_id = f"codex-p0-mysql-{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}

        try:
            checkpointer = get_checkpointer("mysql")
            tables = self._checkpoint_table_collations(checkpointer)
            self.assertTrue(tables)
            self.assertTrue(
                all(collation.startswith("utf8mb4") for collation in tables.values()),
                tables,
            )

            graph = build_application_graph(
                checkpointer=checkpointer,
                supervisor_route_decider=lambda _messages: "business_agent",
                business_node=write_test_state,
            )
            graph.invoke(
                {
                    "messages": [
                        HumanMessage(content="verify mysql checkpoint persistence")
                    ]
                },
                config=config,
            )

            parent_state = graph.get_state(config)
            self.assertEqual(
                parent_state.values["messages"][-1].content,
                "mysql-checkpoint-persistence",
            )

            child_result = self._read_state_in_child_process(thread_id)
            self.assertEqual(child_result["message_count"], 2)
            self.assertEqual(
                child_result["messages"][-1],
                "mysql-checkpoint-persistence",
            )

            checkpointer.delete_thread(thread_id)
            deleted_state = graph.get_state(config)
            self.assertEqual(deleted_state.values.get("messages", []), [])
        finally:
            if checkpointer is not None:
                for cleanup_config in (config,):
                    try:
                        checkpointer.delete_thread(
                            cleanup_config["configurable"]["thread_id"]
                        )
                    except Exception:
                        pass
                close_checkpointer(checkpointer)

    @staticmethod
    def _checkpoint_table_collations(checkpointer) -> dict[str, str]:
        database = os.environ["MYSQL_DATABASE"]
        with checkpointer.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name, table_collation
                FROM information_schema.tables
                WHERE table_schema = %s
                  AND table_name IN (
                    'checkpoints',
                    'checkpoint_blobs',
                    'checkpoint_writes',
                    'checkpoint_migrations'
                  )
                """,
                (database,),
            )
            return {
                str(table_name): str(collation)
                for table_name, collation in cursor.fetchall()
            }

    @staticmethod
    def _read_state_in_child_process(thread_id: str) -> dict:
        script = """
import json
from utils.env import load_environment
from utils.checkpointer import close_checkpointer, get_checkpointer
from graphs.application import build_application_graph

load_environment()
checkpointer = get_checkpointer("mysql")
graph = build_application_graph(
    checkpointer=checkpointer,
    supervisor_route_decider=lambda _messages: "business_agent",
    business_node=lambda _state: {},
)
state = graph.get_state({"configurable": {"thread_id": THREAD_ID}})
messages = state.values.get("messages", []) if state else []
result = {
    "message_count": len(messages),
    "messages": [str(message.content) for message in messages],
}
close_checkpointer(checkpointer)
print("MYSQL_STATE=" + json.dumps(result, ensure_ascii=False))
""".replace("THREAD_ID", json.dumps(thread_id))

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            raise AssertionError(
                "Child process could not restore the MySQL checkpoint:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        marker = "MYSQL_STATE="
        line = next(
            (
                value
                for value in reversed(completed.stdout.splitlines())
                if value.startswith(marker)
            ),
            None,
        )
        if line is None:
            raise AssertionError(
                f"Child process did not return the state marker:\n{completed.stdout}"
            )
        return json.loads(line[len(marker):])


if __name__ == "__main__":
    unittest.main()
