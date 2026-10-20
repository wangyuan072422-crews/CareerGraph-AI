"""
图包。

Studio 使用的图保留为真实子模块，因此 ``graphs.multi_agent_system``
始终只有一个含义。共享图工厂采用延迟导出，
避免在导入包时立即构建图。
"""
from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "multi_agent_system":
        return import_module(f"{__name__}.{name}")
    if name == "build_application_graph":
        from .application import build_application_graph

        return build_application_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["build_application_graph", "multi_agent_system"]
