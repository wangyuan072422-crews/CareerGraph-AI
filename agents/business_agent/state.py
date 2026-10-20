# 业务 Agent 的状态定义。
# 使用 BaseMessage 和 add_messages reducer，
# 正确处理对话消息

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class BusinessAgentState(TypedDict):
    # 带消息历史记录的业务 Agent 状态。

    # 使用 add_messages reducer 可以保证：
    # - 消息追加到列表中
    # - 根据消息 ID 进行适当的去重
    # - 维护对话历史
    messages: Annotated[list[BaseMessage], add_messages]
