"""启动环境加载的确定性测试。"""
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver

from utils import env as env_module


class EnvironmentLoadingTests(unittest.TestCase):
    def test_load_environment_populates_missing_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "CODEX_ENV_TEST_BACKEND=mysql\n"
                "CODEX_ENV_TEST_DATABASE=career\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                loaded = env_module.load_environment(env_path)

                self.assertTrue(loaded)
                self.assertEqual(os.environ["CODEX_ENV_TEST_BACKEND"], "mysql")
                self.assertEqual(os.environ["CODEX_ENV_TEST_DATABASE"], "career")

    def test_process_environment_takes_precedence_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("CODEX_ENV_TEST_BACKEND=mysql\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"CODEX_ENV_TEST_BACKEND": "memory"},
                clear=False,
            ):
                env_module.load_environment(env_path)

                self.assertEqual(os.environ["CODEX_ENV_TEST_BACKEND"], "memory")

    def test_missing_environment_file_is_a_safe_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.env"

            loaded = env_module.load_environment(missing_path)

        self.assertFalse(loaded)

    def test_default_path_resolves_to_project_root_environment_file(self):
        self.assertEqual(
            env_module.DEFAULT_ENV_PATH,
            env_module.PROJECT_ROOT / ".env",
        )


class StartupEntrypointTests(unittest.TestCase):
    def test_api_loads_environment_before_creating_checkpointer(self):
        from api import server
        from utils import checkpointer as checkpointer_module

        events = []
        configured_checkpointer = MemorySaver()

        def record_environment_load():
            events.append("environment")
            return True

        def record_checkpointer_factory(*_args, **_kwargs):
            events.append("checkpointer")
            return configured_checkpointer

        try:
            with (
                patch.object(
                    env_module,
                    "load_environment",
                    side_effect=record_environment_load,
                ),
                patch.object(
                    checkpointer_module,
                    "get_checkpointer",
                    side_effect=record_checkpointer_factory,
                ),
            ):
                reloaded_server = importlib.reload(server)

            self.assertEqual(events[:2], ["environment", "checkpointer"])
            self.assertIs(reloaded_server.checkpointer, configured_checkpointer)
        finally:
            importlib.reload(server)

    def test_studio_entrypoint_loads_environment_before_building_graph(self):
        import graphs.multi_agent_system as studio_entrypoint

        events = []

        def record_environment_load():
            events.append("environment")
            return True

        def record_graph_build():
            events.append("graph")
            return object()

        try:
            with (
                patch.object(
                    env_module,
                    "load_environment",
                    side_effect=record_environment_load,
                ),
                patch(
                    "graphs.application.build_application_graph",
                    side_effect=record_graph_build,
                ),
            ):
                reloaded_entrypoint = importlib.reload(studio_entrypoint)

            self.assertEqual(events, ["environment", "graph"])
            self.assertIsNotNone(reloaded_entrypoint.graph)
        finally:
            importlib.reload(studio_entrypoint)

    def test_main_accepts_mysql_flag(self):
        import main

        args = main._parse_args(["--mysql"])

        self.assertTrue(args.mysql)


if __name__ == "__main__":
    unittest.main()
