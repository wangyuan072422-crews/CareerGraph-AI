"""
State definition for Business Agent.
This uses BaseMessage with add_messages reducer for proper conversation handling.
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class BusinessAgentState(TypedDict):
    """
    State for the business agent with message history.

    Using add_messages reducer ensures:
    - Messages are appended to the list
    - Proper deduplication by message ID
    - Conversation history is maintained
    """
    messages: Annotated[list[BaseMessage], add_messages]
