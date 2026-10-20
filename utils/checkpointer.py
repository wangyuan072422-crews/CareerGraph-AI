"""检查点存储选择与数据库连接配置。

图工厂从调用方接收检查点存储。本模块负责具体的内存/MySQL 实现，
并避免凭据出现在图和 API 代码中。
"""

from __future__ import annotations

import os
from typing import Any, Literal
from urllib.parse import parse_qsl, quote, unquote, urlparse

from langgraph.checkpoint.memory import MemorySaver

# 尝试导入 MySQL 相关依赖，并记录它们是否可用。
try:
    import pymysql
    from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

    MYSQL_AVAILABLE = True
except ImportError:
    pymysql = None
    PyMySQLSaver = None
    MYSQL_AVAILABLE = False


CheckpointerBackend = Literal["memory", "mysql"]


# 创建并返回一个内存检查点存储对象。
def get_memory_saver() -> MemorySaver:
    """创建用于开发和确定性测试的进程内检查点存储。"""
    return MemorySaver()


# 从环境变量取得 MySQL 配置，最后组装成一个数据库连接 URL 字符串。
def build_mysql_connection_string() -> str:
    """从环境变量构建 MySQL URL，同时避免暴露密钥。"""

    mysql_url = os.getenv("MYSQL_URL")
    # os.getenv() 用于读取环境变量。如果已经存在就直接返回
    if mysql_url:
        return mysql_url
    # 没有 MYSQL_URL 时，检查独立配置项
    required_names = (
        "MYSQL_HOST", # 数据库主机地址
        "MYSQL_USER", # 数据库账号
        "MYSQL_PASSWORD",
        "MYSQL_DATABASE", # 数据库名称
    )
    # 那个配置项是空就进入missing列表
    missing = [
        name for name in required_names
        if os.getenv(name) is None
    ]
    # 缺少配置就抛出异常
    if missing:
        raise ValueError(
            "MySQL checkpointer configuration is incomplete. "
            f"Provide MYSQL_URL or set: {', '.join(missing)}."
        )
    # 读取端口 默认使用 3306
    port_value = os.getenv("MYSQL_PORT", "3306")
    try:
        port = int(port_value)
        # 如果不是整数就抛异常
    except ValueError as exc:
        raise ValueError("MYSQL_PORT must be an integer.") from exc

    # 读取账号、密码等内容，并做 URL 编码
    host = os.environ["MYSQL_HOST"]
    user = quote(os.environ["MYSQL_USER"], safe="")
    password = quote(os.environ["MYSQL_PASSWORD"], safe="")
    database = quote(os.environ["MYSQL_DATABASE"], safe="")
    return f"mysql://{user}:{password}@{host}:{port}/{database}"


# 把连接 URL 解析成 pymysql.connect() 能够接收的配置字典。
def _mysql_connection_config(connection_string: str) -> dict[str, Any]:
    """解析支持的 MySQL URL 字段并解码转义后的凭据。"""
    # urlparse() 会把 URL 拆成多个组成部分。
    # 大致可以分为:
    # scheme	mysql
    # username	app_user
    # password	abc%40123
    # hostname	127.0.0.1
    # port	3306
    # path	/agent_db
    parsed = urlparse(connection_string)
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        raise ValueError("MySQL connection string must use the mysql:// scheme.")
    if not parsed.hostname or not parsed.username:
        raise ValueError("MySQL connection string must include a host and user.")
    #
    database = unquote(parsed.path.removeprefix("/"))
    if not database:
        raise ValueError("MySQL connection string must include a database name.")
    # 读取查询参数 URL 除了主机、用户、密码、数据库路径外，还可以带查询参数。
    query = dict(parse_qsl(parsed.query))
    # parse_qsl() 负责把查询字符串解析成键值对，dict() 再把它转换成字典。
    return {
        "host": parsed.hostname,
        "user": unquote(parsed.username),
        "password": unquote(parsed.password or ""),
        "database": database,
        "port": parsed.port or 3306,
        "unix_socket": query.get("unix_socket"),
    }

# build_mysql_connection_string():生产URL字符串
# 而 PyMySQL 的连接函数接受的是关键字参数形式 所以就需要_mysql_connection_config()


# 创建数据库连接、初始化检查点存储、返回存储对象。
def get_mysql_saver(connection_string: str | None = None) -> Any:
    """创建、初始化并返回可用的 MySQL 检查点存储。"""
    # 检查 MySQL 依赖是否可用
    if not MYSQL_AVAILABLE or PyMySQLSaver is None or pymysql is None:
        raise ImportError(
            "MySQL checkpointer is not available. Install it with:\n"
            "  pip install langgraph-checkpoint-mysql pymysql"
        )
    # if connection_string:
    #     resolved_connection_string = connection_string
    # else:
    #     resolved_connection_string = build_mysql_connection_string()
    # 如果调用者明确传入了非空的连接字符串，就优先使用它；
    # 如果没有传入，或者传入的是空字符串，就从环境变量中构建。
    resolved_connection_string = connection_string or build_mysql_connection_string()
    # 把 URL 转换成连接参数 也就是字典
    connection_config = _mysql_connection_config(resolved_connection_string)

    try:
        # 真正连接 MySQL
        connection = pymysql.connect(**connection_config, autocommit=True)
    except Exception as exc:
        raise RuntimeError("Failed to connect to the MySQL checkpointer database.") from exc

    try:
        # 已经建立的 MySQL 连接交给 PyMySQLSaver，创建一个检查点存储对象。
        saver = PyMySQLSaver(connection)
        saver.setup()
    except Exception as exc:
        connection.close()
        raise RuntimeError("Failed to initialize the MySQL checkpointer tables.") from exc

    return saver


# 关闭检查点存储所持有的数据库连接。
def close_checkpointer(checkpointer: Any) -> None:
    """关闭数据库检查点存储，同时不影响 MemorySaver。"""
    # getattr(对象, "属性名", 默认值) 获得这个对象上的属性没有就是默认值
    connection = getattr(checkpointer, "conn", None)
    close_connection = getattr(connection, "close", None)
    # 检查 close_connection 是不是可以被调用的对象；如果是，就执行它。
    if callable(close_connection):
        close_connection()
# 如果 checkpointer 有 conn 属性，并且 checkpointer.conn 有一个可调用的 close 方法，
# 那么最终执行的效果就相当于 checkpointer.conn.close()。


# 这是把前面所有函数串起来的统一入口。
def get_checkpointer(
    backend: CheckpointerBackend | str | None = None,
    *,
    connection_string: str | None = None,
) -> Any:
    """创建已配置的检查点存储。

    ``CHECKPOINTER_BACKEND`` 默认值为 ``memory``，因此本地开发无需数据库。
    需要重启后仍能恢复历史的调用方必须选择 ``mysql``，
    并提供有效的连接配置。
    """
    # 如果调用者传入了一个非空的 backend，使用它；
    # 否则读取 CHECKPOINTER_BACKEND 环境变量；
    # 如果环境变量也不存在，就默认使用 "memory"。
    selected = (backend or os.getenv("CHECKPOINTER_BACKEND", "memory")).strip().lower()
    if selected == "memory":
        return get_memory_saver()
    if selected == "mysql":
        return get_mysql_saver(connection_string)
    raise ValueError(
        "Unsupported checkpointer backend. Expected 'memory' or 'mysql', "
        f"received {selected!r}."
    )


__all__ = [
    "MYSQL_AVAILABLE",
    "build_mysql_connection_string",
    "close_checkpointer",
    "get_checkpointer",
    "get_memory_saver",
    "get_mysql_saver",
]
