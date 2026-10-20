"""
Supervisor Agent 的状态定义。

应用使用统一的共享状态契约，使 Supervisor 和专业 Agent
可以组合在同一张图中。
"""
from graphs.state import ApplicationState


SupervisorState = ApplicationState

__all__ = ["SupervisorState"]
