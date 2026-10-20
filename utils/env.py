"""加载项目环境变量，同时不覆盖已有进程环境变量。"""
# Path 是 Python 标准库 pathlib 中的类，用于以结构化的方式处理文件路径。
from pathlib import Path
# load_dotenv():负责读取 .env 文件，并将其中的配置加载进当前 Python 进程的环境变量。
from dotenv import load_dotenv

# 确定项目路径 __file__ 通常代表当前 Python 模块文件的路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"



def load_environment(
    dotenv_path: str | Path | None = None,
    *,
    override: bool = False,
) -> bool:
    """默认加载项目根目录下的 ``.env``。

    除非显式启用 ``override``，否则已有进程环境变量优先。
    本地默认启动时，文件不存在也不会产生额外操作。
    """
    # if dotenv_path is not None:
    #     path = Path(dotenv_path)
    # else:
    #     path = DEFAULT_ENV_PATH
    # 调用者指定路径，就优先使用指定路径；没有指定，就使用默认的 .env 路径
    path = Path(dotenv_path) if dotenv_path is not None else DEFAULT_ENV_PATH
    return load_dotenv(dotenv_path=path, override=override)


__all__ = ["DEFAULT_ENV_PATH", "PROJECT_ROOT", "load_environment"]
