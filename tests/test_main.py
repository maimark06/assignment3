"""Check command routing without opening or building a real index."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from src.main import main


class MainTests(unittest.TestCase):
    def test_search_command_forwards_index_and_result_limit(self):
        for options, expected_k in (([], 5), (["--top", "3"], 3)):
            with self.subTest(options=options), \
                    patch("src.search_cli.run_search") as run_search:
                main(["search", "--index", "example-index/", *options])
                run_search.assert_called_once_with("example-index/", k=expected_k)

    def test_invalid_search_arguments_do_not_start_console(self):
        for arguments in (
            ["search"],
            ["search", "--index", "index/", "--top", "0"],
            ["search", "--index", "index/", "--top", "-1"],
            ["search", "--index", "index/", "--top", "abc"],
        ):
            with self.subTest(arguments=arguments), \
                    patch("src.search_cli.run_search") as run_search, \
                    redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                main(arguments)
            self.assertEqual(error.exception.code, 2)
            run_search.assert_not_called()

    def test_search_help_lists_options(self):
        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as error:
            main(["search", "--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--index", output.getvalue())
        self.assertIn("--top", output.getvalue())

    def test_existing_build_and_stats_commands_still_route_correctly(self):
        with patch("src.main.run_build") as run_build:
            main(["build", "--data", "data/", "--out", "index/"])
            run_build.assert_called_once_with("data/", "index/", 15000)
        with patch("src.main.run_stats") as run_stats:
            main(["stats", "--index", "index/"])
            run_stats.assert_called_once_with("index/")
