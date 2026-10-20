"""
数据库 Agent：处理来自数据库的结构化数据查询。

模型只会获得一个固定的只读查询工具，不会获得直接执行原始 SQL 的能力。
"""
import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from utils.llm import get_chat_llm

from .state import DatabaseAgentState
from .tools import (
    JobApplicationSearchInput,
    create_job_application_tool,
    search_job_applications,
)


DATABASE_PROMPT = """You are a structured-data assistant for a job-search product.
Use the search_job_applications tool whenever the user asks about saved or
tracked job applications. The tool accepts validated filters, not raw SQL.
Only describe records returned by the tool. Never invent application rows,
statuses, companies, dates, or query results. If the tool reports an error,
explain that live records could not be retrieved. Keep previous conversation
context in mind."""


DatabaseQueryExecutor = Callable[..., dict[str, Any]]


# 把不同形式的模型返回值统一转换为 AIMessage。
def _as_ai_message(response: Any) -> AIMessage:
    """统一模型输出格式，因为测试中注入的模型可能返回字符串。"""
    if isinstance(response, AIMessage):
        return response
    if isinstance(response, BaseMessage):
        return AIMessage(content=str(response.content))
    return AIMessage(content=str(response))


# ：它把 LLM 生成的 tool_calls 转换成真实、受控的数据库查询，
# 再把查询结果转换成 LLM 能继续读取的 ToolMessage。
def _execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    executor: DatabaseQueryExecutor,
) -> list[ToolMessage]:
    """只执行模型生成的求职申请筛选条件，并返回安全的工具消息。"""
    # tool_messages 用来收集所有工具调用结果。
    tool_messages: list[ToolMessage] = []
    # 函数首先创建一个工具，并取得工具名称：
    tool_name = create_job_application_tool().name

    for index, tool_call in enumerate(tool_calls):
        # 正常情况下，模型会提供类似 call_001 的 ID。
        # 如果没提供 就会使用 index来表达id
        tool_call_id = str(tool_call.get("id") or f"database-call-{index}")
        # 这个 ID 很重要，因为最终生成的 ToolMessage
        # 必须通过 tool_call_id 与模型发起的那次工具调用对应起来。
        try:
            # 检查工具名称是否合法
            if tool_call.get("name") != tool_name:
                result = {
                    "ok": False,
                    "error": "The requested database tool is not available.",
                }
            # args：{"status","company","role_keyword","limit"}
            else:
                # 二次校验
                arguments = JobApplicationSearchInput.model_validate(
                    tool_call.get("args") or {}
                )
                # arguments.model_dump() 会把经过校验的 Pydantic 对象转换成字典
                # 然后通过 ** 展开成关键字参数
                # executor 就是：search_job_applications(...)
                result = executor(**arguments.model_dump())
                # 项目规定 executor 必须返回字典。
                if not isinstance(result, dict):
                    result = {
                        "ok": False,
                        "error": "The database tool returned an invalid result.",
                    }
        except Exception:
            # 错误结果 变成 ToolMessage 交给模型。模型可以据此回答：
            result = {
                "ok": False,
                "error": "The business database query failed safely.",
            }
        # 将结果转换成 ToolMessage
        tool_messages.append(
            ToolMessage(
                # json.dumps() 把字典序列化。
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=tool_call_id,
            )
        )

    return tool_messages



def create_database_agent_node(
    llm: Any | None = None,
    query_executor: DatabaseQueryExecutor | None = None,
) -> Callable[[DatabaseAgentState], dict[str, list[BaseMessage]]]:
    """创建可注入模型和查询执行器的数据库节点。"""

    def database_query(state: DatabaseAgentState) -> dict[str, list[BaseMessage]]:
        model = llm or get_chat_llm()
        executor = query_executor or search_job_applications
        prompt_messages = [
            SystemMessage(content=DATABASE_PROMPT),
            *state["messages"],
        ]
        # 先检查模型对象有没有可调用的 bind_tools()。
        bind_tools = getattr(model, "bind_tools", None)
        if not callable(bind_tools):
            return {"messages": [_as_ai_message(model.invoke(prompt_messages))]}

        tool = create_job_application_tool()
        model_with_tools = bind_tools([tool])
        response = model_with_tools.invoke(prompt_messages)
        tool_calls = getattr(response, "tool_calls", None) or []
        # 没有工具调用时直接返回
        if not tool_calls:
            return {"messages": [_as_ai_message(response)]}

        tool_messages = _execute_tool_calls(tool_calls, executor)
        # 模型根据 ToolMessage 中的真实数据库结果生成最终回答
        final_response = model_with_tools.invoke(
            [*prompt_messages, response, *tool_messages]
        )
        # 第二次模型调用后，如果模型又要求调用工具，
        # 代码不会继续执行第二轮查询，而是返回固定安全回答。
        if getattr(final_response, "tool_calls", None):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I could not safely complete the database request "
                            "in one tool round."
                        )
                    )
                ]
            }
        return {"messages": [_as_ai_message(final_response)]}

    return database_query


database_query_node = create_database_agent_node()


def create_database_agent_graph(
    llm: Any | None = None,
    query_executor: DatabaseQueryExecutor | None = None,
):
    """创建数据库 Agent 图。"""
    workflow = StateGraph(DatabaseAgentState)

    workflow.add_node(
        "database_query",
        create_database_agent_node(llm=llm, query_executor=query_executor),
    )

    workflow.add_edge(START, "database_query")
    workflow.add_edge("database_query", END)

    return workflow
