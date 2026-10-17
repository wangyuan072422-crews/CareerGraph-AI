"""
State definition for Supervisor Agent.
"""
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class SupervisorState(TypedDict):
    """
    State for the supervisor agent.
    Includes routing decision and message history.
    """
    # messages →   类.型 list[BaseMessage]，合并规则 = add_messages （合并）
    messages: Annotated[list[BaseMessage], add_messages]
    next: str  # Which agent to route to next
