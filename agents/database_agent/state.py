"""
State definition for Database Agent.
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class DatabaseAgentState(TypedDict):
    """
    State for the database agent with message history.
    """
    messages: Annotated[list[BaseMessage], add_messages]
