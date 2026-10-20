"""检查点存储后端选择与配置的确定性测试。"""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver

from utils import checkpointer as checkpointer_module


class FakePyMySQLSaver:
    instances = []

    def __init__(self, connection):
        self.connection = connection
        self.setup_calls = 0
        self.__class__.instances.append(self)

    def setup(self):
        self.setup_calls += 1


class FakeConnection:
    def __init__(self, *, setup_error=None):
        self.close_calls = 0
        self.setup_error = setup_error

    def close(self):
        self.close_calls += 1


class FakePyMySQL:
    def __init__(self, connection=None, connect_error=None):
        self.connection = connection or FakeConnection()
        self.connect_error = connect_error
        self.connect_calls = []

    def connect(self, **kwargs):
        self.connect_calls.append(kwargs)
        if self.connect_error is not None:
            raise self.connect_error
        return self.connection


class CheckpointerConfigurationTests(unittest.TestCase):
    def test_mysql_dependency_is_importable(self):
        self.assertTrue(checkpointer_module.MYSQL_AVAILABLE)
        self.assertTrue(callable(checkpointer_module.PyMySQLSaver))
        self.assertIsNotNone(checkpointer_module.pymysql)

    def test_memory_backend_returns_memory_saver(self):
        with patch.dict(os.environ, {}, clear=True):
            saver = checkpointer_module.get_checkpointer()

        self.assertIsInstance(saver, MemorySaver)

    def test_mysql_backend_uses_explicit_connection_string(self):
        connection_string = (
            "mysql://career%20user:p%40ss%20word@db.internal:3307/career_db"
            "?unix_socket=%2Ftmp%2Fmysql.sock"
        )
        connection = FakeConnection()
        fake_pymysql = FakePyMySQL(connection=connection)
        FakePyMySQLSaver.instances = []

        with (
            patch.object(checkpointer_module, "MYSQL_AVAILABLE", True),
            patch.object(checkpointer_module, "PyMySQLSaver", FakePyMySQLSaver),
            patch.object(checkpointer_module, "pymysql", fake_pymysql),
        ):
            saver = checkpointer_module.get_checkpointer(
                "mysql",
                connection_string=connection_string,
            )

        self.assertIsInstance(saver, FakePyMySQLSaver)
        self.assertIs(saver.connection, connection)
        self.assertEqual(saver.setup_calls, 1)
        self.assertEqual(
            fake_pymysql.connect_calls,
            [
                {
                    "host": "db.internal",
                    "user": "career user",
                    "password": "p@ss word",
                    "database": "career_db",
                    "port": 3307,
                    "unix_socket": "/tmp/mysql.sock",
                    "autocommit": True,
                }
            ],
        )

    def test_mysql_setup_failure_closes_connection_and_wraps_error(self):
        connection = FakeConnection()
        fake_pymysql = FakePyMySQL(connection=connection)

        class BrokenSaver(FakePyMySQLSaver):
            def setup(self):
                raise RuntimeError("migration failed")

        with (
            patch.object(checkpointer_module, "MYSQL_AVAILABLE", True),
            patch.object(checkpointer_module, "PyMySQLSaver", BrokenSaver),
            patch.object(checkpointer_module, "pymysql", fake_pymysql),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "initialize the MySQL checkpointer tables",
            ):
                checkpointer_module.get_mysql_saver(
                    "mysql://user:pass@localhost:3306/career",
                )

        self.assertEqual(connection.close_calls, 1)

    def test_mysql_connect_failure_has_clear_error(self):
        fake_pymysql = FakePyMySQL(
            connect_error=ConnectionError("database refused connection"),
        )

        with (
            patch.object(checkpointer_module, "MYSQL_AVAILABLE", True),
            patch.object(checkpointer_module, "PyMySQLSaver", FakePyMySQLSaver),
            patch.object(checkpointer_module, "pymysql", fake_pymysql),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "connect to the MySQL checkpointer database",
            ):
                checkpointer_module.get_mysql_saver(
                    "mysql://user:pass@localhost:3306/career",
                )

    def test_close_checkpointer_closes_database_connection(self):
        connection = FakeConnection()

        checkpointer_module.close_checkpointer(
            SimpleNamespace(conn=connection),
        )

        self.assertEqual(connection.close_calls, 1)

    def test_close_checkpointer_ignores_memory_saver(self):
        checkpointer_module.close_checkpointer(MemorySaver())

    def test_mysql_connection_is_built_and_escaped_from_environment(self):
        environment = {
            "MYSQL_HOST": "db.internal",
            "MYSQL_PORT": "3307",
            "MYSQL_USER": "career user",
            "MYSQL_PASSWORD": "p@ss word",
            "MYSQL_DATABASE": "career_db",
        }

        with patch.dict(os.environ, environment, clear=True):
            connection_string = checkpointer_module.build_mysql_connection_string()

        self.assertEqual(
            connection_string,
            "mysql://career%20user:p%40ss%20word@db.internal:3307/career_db",
        )

    def test_mysql_url_environment_value_takes_precedence(self):
        environment = {
            "MYSQL_URL": "mysql://override:secret@db:3306/override_db",
            "MYSQL_HOST": "ignored",
        }

        with patch.dict(os.environ, environment, clear=True):
            connection_string = checkpointer_module.build_mysql_connection_string()

        self.assertEqual(
            connection_string,
            "mysql://override:secret@db:3306/override_db",
        )

    def test_missing_mysql_configuration_has_clear_error(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(checkpointer_module, "MYSQL_AVAILABLE", True),
            patch.object(checkpointer_module, "PyMySQLSaver", FakePyMySQLSaver),
            patch.object(checkpointer_module, "pymysql", FakePyMySQL()),
        ):
            with self.assertRaisesRegex(ValueError, "MYSQL_HOST"):
                checkpointer_module.get_mysql_saver()

    def test_missing_mysql_dependency_has_install_hint(self):
        with patch.object(checkpointer_module, "MYSQL_AVAILABLE", False):
            with self.assertRaisesRegex(ImportError, "langgraph-checkpoint-mysql"):
                checkpointer_module.get_mysql_saver("mysql://user:pass@host/db")

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "memory.*mysql"):
            checkpointer_module.get_checkpointer("sqlite")


if __name__ == "__main__":
    unittest.main()
