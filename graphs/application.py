"""
FastAPI、Studio 和测试共享的单一应用图。

该图负责路由和错误边界。检查点存储的创建仍由调用方负责，
因此 Studio 和 FastAPI 可以提供不同的持久化方式。
"""
import logging
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from agents.business_agent import create_business_agent_node
from agents.database_agent import create_database_agent_node
from agents.supervisor import RouteDecision, create_supervisor_node
from graphs.state import ApplicationState


logger = logging.getLogger(__name__)
# 表示节点返回的 State 局部更新
StateUpdate = dict[str, Any]
# 表示节点函数的类型：输入：ApplicationState 输出：StateUpdate 字典
StateNode = Callable[[ApplicationState], StateUpdate]
# 这是发生错误时展示给用户的安全文案，不会暴露异常堆栈或数据库连接信息。
SAFE_ERROR_MESSAGE = (
    "I could not complete this request safely. Please try again. "
    "If the problem continues, start a new conversation."
)
# 当 Supervisor 判断不需要专业 Agent 时，用这条消息作为本轮回答。
FINISH_MESSAGE = "No further specialist action is needed for this request."


# guarded = 先尝试执行传来的节点 正常就返回它的结果 不正常就转换成统一的错误状态
# name:节点的名字，用于日志和错误信息,node：真正的节点函数
# 给节点增加统一异常保护，并补齐 status、error字段
def _guarded_node(name: str, node: StateNode) -> StateNode:
    """将节点异常转换为可预测的状态更新。"""

    def guarded(state: ApplicationState) -> StateUpdate:
        try:
            # 执行真正的节点 同时节点产生的结果(dict)就是这个update
            # 也有可能结果是None/{}
            update = node(state) or {}
        # 节点运行出现异常 就进入异常分支
        except Exception as exc:
            logger.exception("Application node %s failed", name)
            # 将异常转换成状态更新
            return {
                "next": "error",
                "status": "error", # 告诉 API 和调用方，本次执行失败
                # 保存内部技术错误，供日志和调试使用
                "error": f"{name}:{type(exc).__name__}: {exc}",
                # 说明节点在产生回答之前失败了
                "route_reason": f"{name} failed before producing a response.",
            }
        # 将节点返回的字典复制一份 避免直接修改原字典对象
        normalized = dict(update)
        # 第一层 state.get("status", "ok")：从当前 State 字典读取 status。假设是degraded就返回这个，没有就默认是OK
        # 第二层 ormalized.setdefault("status", 上面读取的值)
        # 如果字典有status 就保留节点返回的status 没有就使用刚刚state读取的值
        normalized.setdefault("status", state.get("status", "ok"))
        normalized.setdefault("error", state.get("error"))
        return normalized

    return guarded


# Supervisor 执行后，根据 State 决定进入哪个节点
def _route_after_supervisor(
    state: ApplicationState,
) -> Literal["business_agent", "database_agent", "finish_handler", "error_handler"]:
    if state.get("status") == "error":
        return "error_handler"

    next_node = state.get("next")
    if next_node == "business_agent":
        return "business_agent"
    if next_node == "database_agent":
        return "database_agent"
    if next_node == "finish":
        return "finish_handler"
    return "error_handler"


# 专业 Agent 执行后，决定正常结束还是进入错误节点
def _route_after_agent(state: ApplicationState) -> Literal["error_handler", "__end__"]:
    if state.get("status") == "error":
        return "error_handler"
    return END


# 当不需要专业 Agent 时，生成一条结束消息
def _finish_handler(state: ApplicationState) -> StateUpdate:
    return {
        "messages": [AIMessage(content=FINISH_MESSAGE)],
        "next": "finish",
        "status": state.get("status", "ok"),
        "error": state.get("error"),
    }


# 生成安全错误消息，并结束异常流程
def _error_handler(state: ApplicationState) -> StateUpdate:
    return {
        "messages": [AIMessage(content=SAFE_ERROR_MESSAGE)],
        "next": "error",
        "status": "error",
        "route_reason": "The request stopped at the application error boundary.",
    }


def build_application_graph(
    checkpointer: Any | None = None,
    *,
    supervisor_llm: Any | None = None,
    supervisor_route_decider: Callable[[list[Any]], RouteDecision | dict[str, Any] | str] | None = None,
    business_llm: Any | None = None,
    database_llm: Any | None = None,
    business_node: StateNode | None = None,
    database_node: StateNode | None = None,
):
    """
    构建可执行的应用图。

    LLM 和节点参数用于确定性测试以及后续工具注入。不传入检查点存储时，
    持久化由调用方负责；当 LangGraph Studio 管理运行时检查点存储时，
    必须采用这种方式。
    """
    supervisor = create_supervisor_node(
        llm=supervisor_llm,
        route_decider=supervisor_route_decider,
    )
    # 业务节点的调用关系：business(state) = create_business_agent_node(llm=business_llm) = business_query(state)
    business = business_node or create_business_agent_node(llm=business_llm)
    database = database_node or create_database_agent_node(llm=database_llm)

    workflow = StateGraph(ApplicationState)
    workflow.add_node("supervisor", _guarded_node("supervisor", supervisor))
    workflow.add_node("business_agent", _guarded_node("business_agent", business))
    workflow.add_node("database_agent", _guarded_node("database_agent", database))
    workflow.add_node("finish_handler", _guarded_node("finish_handler", _finish_handler))
    workflow.add_node("error_handler", _error_handler)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "business_agent": "business_agent",
            "database_agent": "database_agent",
            "finish_handler": "finish_handler",
            "error_handler": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "business_agent",
        _route_after_agent,
        {
            "error_handler": "error_handler",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "database_agent",
        _route_after_agent,
        {
            "error_handler": "error_handler",
            END: END,
        },
    )
    workflow.add_edge("finish_handler", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile(checkpointer=checkpointer, name="application_graph")


__all__ = ["build_application_graph"]
