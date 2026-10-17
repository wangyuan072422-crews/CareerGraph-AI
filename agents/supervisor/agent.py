"""
Supervisor Agent - Routes queries to appropriate specialized agents.
"""
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils.llm import get_chat_llm
from langgraph.graph import StateGraph, START, END
from .state import SupervisorState
from typing import Literal


def supervisor_node(state: SupervisorState) -> SupervisorState:
    """
    Analyze the query and route to appropriate agent.
    """
    # 从状态中取出当前积累的全部对话
    messages = state["messages"]

    # ChatOpenAI(...) 创建一个客户端实例
    llm = get_chat_llm("qwen3:4b")
    # 它读对话、按职责表判断意图、只吐出一个路由标签，代码拿这个标签决定下一步走向。
    system_prompt = """You are a supervisor routing queries to specialized agents.

    Available agents:
    - business_agent: Handles questions about business processes, policies, KB documents, supply chain operations
    - database_agent: Handles queries about structured data, SQL queries, database information
    - FINISH: Use this when the query has been fully answered

    Analyze the user's query and respond with ONLY the agent name: business_agent, database_agent, or FINISH.
    Consider the conversation history to maintain context."""

    # 分类消息
    routing_messages = [
        SystemMessage(content=system_prompt),
        # 告诉LLM：根据这段对话，该哪个agent处理 并把被测的对话内容{message}当作题目交给他
        HumanMessage(content=f"Based on this conversation history, which agent should handle the query? Respond with ONLY: business_agent, database_agent, or FINISH.\n\nConversation: {messages}")
    ]
    # 返回的是AIMessage
    response = llm.invoke(routing_messages)

    # E取文本，去空白、转小写
    next_agent = response.content.strip().lower()

    # Validate routing decision
    if next_agent not in ["business_agent", "database_agent", "finish"]:
        next_agent = "business_agent"  # Default to business agent

    # 返回的是一个部分更新的dict 不是完整state
    return {
        "next": next_agent,
        "messages": [AIMessage(content=f"Routing to: {next_agent}")]
    }


def route_to_agent(state: SupervisorState) -> Literal["business_agent", "database_agent", "__end__"]:
    """
    Conditional edge function to route to the appropriate agent.
    """
    if state["next"] == "finish":
        return "__end__"
    return state["next"]


def create_supervisor_graph():
    """
    Create the supervisor graph with routing logic.
    """
    workflow = StateGraph(SupervisorState)

    # Add supervisor node
    workflow.add_node("supervisor", supervisor_node)

    # Add edge from start to supervisor
    workflow.add_edge(START, "supervisor")

    # Add conditional edges for routing
    # Note: In a full implementation, you would call the sub-agents here
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "business_agent": END,  # In full impl, this would route to business_agent node
            "database_agent": END,  # In full impl, this would route to database_agent node
            "__end__": END
        }
    )

    return workflow
