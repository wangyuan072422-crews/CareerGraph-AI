"""
直接使用 Python 运行时，演示检查点存储正确用法的主入口。

重要提示 - 部署场景：
此文件用于直接通过 Python 运行 Agent，而不是使用 langgraph dev/Studio。
直接运行时，必须向 compile() 传入检查点存储。

此文件：直接使用 Python 运行 → 使用检查点存储
graphs/*.py：LangGraph Studio → 不传入检查点存储
  
关键点：
1. 调用时在线程配置中传入 thread ID，而不是放进状态
2. 直接使用 Python 时，必须传入检查点存储（MemorySaver 或 MySQL）
3. 使用 LangGraph Studio 时，参见 graphs/multi_agent_system.py（不传入检查点存储）
"""
import argparse
import uuid

from langchain_core.messages import HumanMessage

from utils.checkpointer import get_checkpointer
from utils.env import load_environment

# 建一张业务图 → 给它配一个 checkpointer → 用同一个 thread_id 连问三句 → 打印回答，顺便演示"换个 thread_id 就失忆"。

def run_agent_with_memory(use_mysql: bool = False):
    """
    演示如何正确使用检查点存储维护多轮对话记忆。

    参数：
        use_mysql：为 True 时使用 MySQL 检查点存储；
        为 False 时使用内存检查点存储。
    """
    load_environment()
    from graphs.application import build_application_graph

    backend = "mysql" if use_mysql else "memory"
    print(f"Using {backend} Checkpointer")
    checkpointer = get_checkpointer(backend)

    graph = build_application_graph(checkpointer=checkpointer)

    # 为本次对话创建线程 ID。
    # 重要：线程 ID 放在配置中，而不是状态中。
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n=== Starting conversation with thread_id: {thread_id} ===\n")

    # 第一次查询。
    print("Query 1: How should I structure a behavioral interview answer?")
    result1 = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="How should I structure a behavioral interview answer?"
                )
            ],
            "forced_agent": None,
            "status": "ok",
            "error": None,
        },
        config=config
    )
    print(f"Response: {result1['messages'][-1].content}\n")

    # 后续查询：这里应当保留上下文。
    print("Query 2: Can you give me an example of the second step?")
    result2 = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Can you give me an example of the second step?"
                )
            ],
            "forced_agent": None,
            "status": "ok",
            "error": None,
        },
        config=config
    )
    print(f"Response: {result2['messages'][-1].content}\n")

    # 再一次后续查询，用于验证上下文能够保持。
    print("Query 3: How long should that answer be?")
    result3 = graph.invoke(
        {
            "messages": [HumanMessage(content="How long should that answer be?")],
            "forced_agent": None,
            "status": "ok",
            "error": None,
        },
        config=config
    )
    print(f"Response: {result3['messages'][-1].content}\n")

    print("=== Conversation complete ===")
    print(f"All messages maintained context using thread_id: {thread_id}")
    # 返回带记忆的图对象 和 刚才用的id
    return graph, thread_id

# 换一个新thread_id 问一个和第三次一模一样的问题
def demonstrate_new_conversation(graph, old_thread_id):
    """
    演示新的线程 ID 会开启一段全新对话。
    """
    new_thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": new_thread_id}}

    print(f"\n=== New conversation with thread_id: {new_thread_id} ===\n")
    print("Query: How long should that answer be?")
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="How long should that answer be?")],
            "forced_agent": None,
            "status": "ok",
            "error": None,
        },
        config=config
    )
    print(f"Response: {result['messages'][-1].content}\n")
    print("Notice: The agent doesn't have context because it's a new thread_id")


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run the multi-agent career assistant directly."
    )
    parser.add_argument(
        "--mysql",
        action="store_true",
        help="Use the environment-configured MySQL checkpointer.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    graph, thread_id = run_agent_with_memory(use_mysql=args.mysql)

    # 演示新对话。
    demonstrate_new_conversation(graph, thread_id)

    print("\n" + "="*60)
    print("To use MySQL checkpointer:")
    print("1. Create .env from .env.example")
    print("2. Set CHECKPOINTER_BACKEND=mysql and valid MYSQL_* credentials")
    print("3. Run: python main.py --mysql")
    print("="*60)


if __name__ == "__main__":
    main()
