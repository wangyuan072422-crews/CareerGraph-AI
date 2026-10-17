"""
Database Agent - Handles structured data queries from databases.
"""
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm import get_chat_llm
from langgraph.graph import StateGraph, START, END
from .state import DatabaseAgentState


def database_query_node(state: DatabaseAgentState) -> DatabaseAgentState:
    """
    Process database-related queries.
    Simulates SQL query generation and execution.
    """
    messages = state["messages"]

    llm = get_chat_llm("qwen3:4b")

    # 该 Agent 可以查询 SQL 数据库、生成 SQL、解释结果
    system_prompt = """You are a database query assistant.
    You help users query structured data from SQL databases.
    You can generate SQL queries and explain the results.
    Maintain context from previous messages in the conversation."""

    full_messages = [
        {"role": "system", "content": system_prompt}
    ] + messages

    response = llm.invoke(full_messages)

    return {"messages": [response]}


def create_database_agent_graph():
    """Create the database agent graph."""
    workflow = StateGraph(DatabaseAgentState)

    workflow.add_node("database_query", database_query_node)

    workflow.add_edge(START, "database_query")
    workflow.add_edge("database_query", END)

    return workflow
