"""图包导入接口的测试。"""
import inspect
import subprocess
import sys
import unittest
from pathlib import Path

import graphs
from graphs import build_application_graph, multi_agent_system
from graphs.application import build_application_graph as application_graph_factory


class GraphExportTests(unittest.TestCase):
    def test_bare_package_attribute_loads_the_module_not_the_graph(self):
        script = (
            "import graphs\n"
            "value = getattr(graphs, 'multi_agent_system')\n"
            "print(type(value).__name__)\n"
            "print(value.__name__)\n"
            "print(type(value.graph).__name__)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout.strip().splitlines()
        self.assertEqual(output[0], "module")
        self.assertEqual(output[1], "graphs.multi_agent_system")
        self.assertEqual(output[2], "CompiledStateGraph")

    def test_multi_agent_system_is_the_real_module(self):
        self.assertTrue(inspect.ismodule(multi_agent_system))
        self.assertEqual(multi_agent_system.__name__, "graphs.multi_agent_system")
        self.assertIs(graphs.multi_agent_system, multi_agent_system)
        self.assertTrue(callable(multi_agent_system.graph.invoke))

    def test_build_application_graph_is_exported_as_a_function(self):
        self.assertIs(build_application_graph, application_graph_factory)
        self.assertFalse(inspect.ismodule(build_application_graph))

    def test_star_import_keeps_multi_agent_system_as_a_module(self):
        namespace = {}

        exec("from graphs import *", namespace)

        self.assertIs(namespace["multi_agent_system"], multi_agent_system)
        self.assertIs(namespace["build_application_graph"], build_application_graph)


if __name__ == "__main__":
    unittest.main()
