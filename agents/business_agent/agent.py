"""
Business Agent - Handles queries about business documents and KB.
This agent demonstrates proper conversational memory with checkpointer.
"""
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm import get_chat_llm
from langgraph.graph import StateGraph, START, END
from .state import BusinessAgentState


def business_query_node(state: BusinessAgentState) -> BusinessAgentState:
    """
    Process business-related queries.
    The LLM has access to full conversation history via state["messages"].
    """
    # Get the conversation history
    messages = state["messages"]

    # Initialize LLM
    llm = get_chat_llm("qwen3:4b")

    # System message to set context
    system_prompt = """You are a helpful business assistant with access to company knowledge base documents.
    You can answer questions about business processes, policies, and supply chain operations.
    Maintain context from previous messages in the conversation."""

    # Create messages with system prompt
    full_messages = [
        {"role": "system", "content": system_prompt} # 里面是dict 但{...}是列表
    ] + messages # 列表

    # Get response from LLM
    response = llm.invoke(full_messages)

    # Return updated state with new message
    # The add_messages reducer will append this to the history
    return {"messages": [response]}


def create_business_agent_graph():
    """
    Create the business agent graph.

    IMPORTANT: Do NOT compile with checkpointer here.
    The checkpointer should be passed when compiling at the top level.
    """
    # Create graph
    workflow = StateGraph(BusinessAgentState)

    # Add nodes
    workflow.add_node("business_query", business_query_node)

    # Add edges
    workflow.add_edge(START, "business_query")
    workflow.add_edge("business_query", END)

    # Return the workflow WITHOUT compiling
    # This allows the parent to compile with appropriate checkpointer
    return workflow
