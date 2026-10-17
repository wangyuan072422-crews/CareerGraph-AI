"""
Checkpointer configuration for conversational memory.

IMPORTANT NOTES:
1. For InMemorySaver: DO NOT pass checkpointer to compile() - it's built-in
2. For PyMySQLSaver: You MUST pass checkpointer to compile()
3. Thread ID should be passed in config during invocation, NOT in the state
"""
from langgraph.checkpoint.memory import MemorySaver

# MySQL checkpointer is optional - only import if installed
try:
    from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    PyMySQLSaver = None


# 每调用一次，就创建一个新的内存存储对象。
# 也就是用来生成checkpoint
def get_memory_saver():
    """
    Get InMemorySaver for testing.

    NOTE: When using InMemorySaver, you don't need to pass it to compile().
    LangGraph automatically uses it when no checkpointer is provided.

    However, if you want to share memory across multiple graph instances,
    you should create one and pass it explicitly.
    """
    return MemorySaver()


def get_mysql_saver(connection_string: str = None):
    """
    Get MySQL checkpointer for production use.

    Args:
        connection_string: MySQL connection string in format:
            "mysql://user:password@host:port/database"

    Example:
        checkpointer = get_mysql_saver("mysql://user:pass@localhost:3306/langgraph")
        graph = workflow.compile(checkpointer=checkpointer)

    IMPORTANT: You MUST pass this checkpointer to compile() for it to work.

    Raises:
        ImportError: If langgraph-checkpoint-mysql is not installed
    """
    # 看看有没有这个依赖 没有就中断执行 告诉开发者怎么安装
    if not MYSQL_AVAILABLE:
        raise ImportError(
            "MySQL checkpointer is not available. Install it with:\n"
            "  pip install langgraph-checkpoint-mysql pymysql"
        )

    if not connection_string:
        # Default for testing
        connection_string = "mysql://root:password@localhost:3306/langgraph_db"

    return PyMySQLSaver.from_conn_string(connection_string)


# Configuration examples
CHECKPOINTER_CONFIG = {
    "memory": {
        "description": "In-memory checkpointer for development/testing",
        "usage": """
        # Option 1: Don't pass anything (uses built-in MemorySaver)
        graph = workflow.compile()

        # Option 2: Create and pass explicitly for shared memory
        checkpointer = MemorySaver()
        graph = workflow.compile(checkpointer=checkpointer)
        """
    },
    "mysql": {
        "description": "MySQL checkpointer for production",
        "usage": """
        from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

        # Create MySQL checkpointer
        checkpointer = PyMySQLSaver.from_conn_string(
            "mysql://user:password@host:port/database"
        )

        # MUST pass to compile()
        graph = workflow.compile(checkpointer=checkpointer)
        """
    }
}


# get_memory_saver()
#   -> 创建一个内存版 checkpointer
#   -> 使用 MemorySaver / InMemorySaver
#   -> 快照保存在当前 Python 进程内存中
#   -> 服务重启后数据丢失
#
# get_mysql_saver()
#   -> 创建一个 MySQL 版 checkpointer
#   -> 使用 PyMySQLSaver
#   -> 快照保存在 MySQL 数据库中
#   -> 服务重启后仍可恢复