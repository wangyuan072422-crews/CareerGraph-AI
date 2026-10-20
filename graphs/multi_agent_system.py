"""
LangGraph Studio 入口。

Studio 负责持久化，因此编译共享应用图时不传入检查点存储。
FastAPI 和直接使用 Python 的调用方则向同一个图工厂传入各自的
检查点存储。
"""
from utils.env import load_environment

load_environment()

from graphs.application import build_application_graph


graph = build_application_graph()

__all__ = ["graph"]
