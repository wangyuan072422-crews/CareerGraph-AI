"""数据库 Agent 只读工具调用的确定性测试。"""
import json
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from agents.database_agent import (
    JobApplicationSearchInput,
    create_database_agent_node,
    search_job_applications,
)
from agents.database_agent import tools as database_tools


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.execute_calls = []

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def execute(self, sql, parameters):
        self.execute_calls.append((sql, parameters))

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.cursor_instance = FakeCursor(rows)
        self.close_calls = 0

    def cursor(self):
        return self.cursor_instance

    def close(self):
        self.close_calls += 1


class ToolCallingChatModel:
    def __init__(self, tool_args):
        self.tool_args = tool_args
        self.bound_tools = []
        self.calls = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_job_applications",
                        "args": self.tool_args,
                        "id": "call-1",
                    }
                ],
            )
        return AIMessage(content="You have one applied backend role at Acme.")


class DatabaseAgentToolTests(unittest.TestCase):
    def test_search_uses_fixed_sql_and_parameterized_filters(self):
        connection = FakeConnection(
            [
                {
                    "id": 7,
                    "company": "Acme",
                    "role_title": "Backend Intern",
                    "status": "applied",
                }
            ]
        )
        malicious_company = "Acme%_Corp' OR 1=1 --"
        environment = {
            "BUSINESS_DB_HOST": "localhost",
            "BUSINESS_DB_USER": "career_app",
            "BUSINESS_DB_PASSWORD": "secret",
            "BUSINESS_DB_DATABASE": "career_assistant",
        }

        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                database_tools,
                "_connect_business_database",
                return_value=connection,
            ),
        ):
            result = search_job_applications(
                status="applied",
                company=malicious_company,
                role_keyword="Backend",
                limit=5,
            )

        sql, parameters = connection.cursor_instance.execute_calls[0]
        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 1)
        self.assertIn("SELECT id, company, role_title, status, applied_at", sql)
        self.assertIn("status = %s", sql)
        self.assertIn("company LIKE %s ESCAPE '/'", sql)
        self.assertNotIn(malicious_company, sql)
        self.assertEqual(parameters[0], "applied")
        self.assertEqual(parameters[1], "%Acme/%/_Corp' OR 1=1 --%")
        self.assertEqual(parameters[-1], 5)
        self.assertEqual(connection.close_calls, 1)

    def test_invalid_status_is_rejected_before_query_execution(self):
        with self.assertRaises(ValidationError):
            JobApplicationSearchInput(status="anything", limit=10)

        result = search_job_applications(status="anything")
        self.assertFalse(result["ok"])
        self.assertIn("invalid", result["error"])

    def test_unconfigured_database_returns_safe_error(self):
        with patch.dict(os.environ, {}, clear=True):
            result = search_job_applications(limit=5)

        self.assertFalse(result["ok"])
        self.assertIn("not configured", result["error"])

    def test_database_node_executes_tool_and_keeps_trace_out_of_graph_history(self):
        executor_calls = []

        def fake_executor(**filters):
            executor_calls.append(filters)
            return {
                "ok": True,
                "count": 1,
                "rows": [
                    {
                        "company": "Acme",
                        "role_title": "Backend Intern",
                        "status": "applied",
                    }
                ],
            }

        model = ToolCallingChatModel(
            {"status": "applied", "company": "Acme", "limit": 3}
        )
        node = create_database_agent_node(
            llm=model,
            query_executor=fake_executor,
        )

        result = node({"messages": [HumanMessage(content="Show my Acme applications")]})

        self.assertEqual(executor_calls[0]["status"], "applied")
        self.assertEqual(executor_calls[0]["company"], "Acme")
        self.assertEqual(len(model.bound_tools), 1)
        self.assertIsInstance(model.calls[1][-1], ToolMessage)
        tool_payload = json.loads(model.calls[1][-1].content)
        self.assertTrue(tool_payload["ok"])
        self.assertEqual(tool_payload["rows"][0]["company"], "Acme")
        self.assertEqual(len(result["messages"]), 1)
        self.assertIsInstance(result["messages"][0], AIMessage)
        self.assertIn("Acme", result["messages"][0].content)

    def test_tool_exception_is_returned_as_safe_message(self):
        def broken_executor(**_filters):
            raise RuntimeError("password should not leak")

        model = ToolCallingChatModel({"company": "Acme"})
        node = create_database_agent_node(
            llm=model,
            query_executor=broken_executor,
        )

        node({"messages": [HumanMessage(content="Show my applications")]})

        tool_message = model.calls[1][-1]
        self.assertIsInstance(tool_message, ToolMessage)
        self.assertIn("failed safely", tool_message.content)
        self.assertNotIn("password should not leak", tool_message.content)


if __name__ == "__main__":
    unittest.main()
