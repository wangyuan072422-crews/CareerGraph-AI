# Database Agent 的状态定义。

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class DatabaseAgentState(TypedDict):

    # 带消息历史记录的 Database Agent 状态。

    messages: Annotated[list[BaseMessage], add_messages]
