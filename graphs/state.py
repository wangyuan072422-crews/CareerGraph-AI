"""
应用图共享的状态契约。

检查点存储会保存此状态。thread_id 等请求级配置属于调用配置，
不应放入此状态。
"""
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


RouteName = Literal["business_agent", "database_agent", "finish"]
ExecutionStatus = Literal["ok", "degraded", "error"]


class ApplicationState(TypedDict):
    """Supervisor 和所有专业 Agent 共享的状态。"""

    # 消息历史字段 add_messages合并 不是覆盖
    messages: Annotated[list[BaseMessage], add_messages]
    # Supervisor选择的下一个节点
    # business_agent、database_agent、finish 或 error。
    next: NotRequired[RouteName | Literal["error"]]
    # 路由原因，供调试和 API 返回使用，不写入消息历史。
    route_reason: NotRequired[str]
    # OK：正常完成 Supervisor选择了agent 业务agent也正常返回来回答
    # degraded:Supervisor 返回了非法或无法解析的路由->系统不敢让请求直接失败->自动降级到 business_agent->status = degraded
    # error:执行失败 无法正常完成
    status: NotRequired[ExecutionStatus]
    error: NotRequired[str | None]
    forced_agent: NotRequired[RouteName | None]
