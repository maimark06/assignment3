"""Exercise console behavior before the real Searcher is available."""

from contextlib import redirect_stdout
from io import StringIO
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.search_cli import run_search


class SearchCliTests(unittest.TestCase):
    def test_search_output_empty_input_and_exit(self):
        module = ModuleType("src.searcher")
        searcher = Mock()
        searcher.search.side_effect = [
            ([SimpleNamespace(url="https://example.com/acm", score=1.25)], 7),
            ([], 0),
        ]
        context = Mock()
        context.__enter__ = Mock(return_value=searcher)
        context.__exit__ = Mock(return_value=False)
        module.Searcher = Mock(return_value=context)
        output = StringIO()

        with patch.dict("sys.modules", {"src.searcher": module}), \
                patch("builtins.input", side_effect=["  ", " ACM ", "missing", " EXIT "]), \
                patch("src.search_cli.perf_counter", side_effect=[10, 10.025, 20, 20.01]), \
                redirect_stdout(output):
            run_search("index/", k=3)

        module.Searcher.assert_called_once_with("index/")
        self.assertEqual(searcher.search.call_count, 2)
        searcher.search.assert_any_call("ACM", k=3)
        searcher.search.assert_any_call("missing", k=3)
        self.assertIn("Matches: 7 | Search time: 25.00 ms", output.getvalue())
        self.assertIn("1. https://example.com/acm (score: 1.250000)", output.getvalue())
        self.assertIn("Matches: 0", output.getvalue())
        self.assertIn("No results found.", output.getvalue())
        context.__exit__.assert_called_once()

    def test_quit_eof_and_interrupt_close_searcher(self):
        for ending in ("quit", EOFError(), KeyboardInterrupt()):
            with self.subTest(ending=ending):
                module = ModuleType("src.searcher")
                context = Mock()
                context.__enter__ = Mock(return_value=Mock())
                context.__exit__ = Mock(return_value=False)
                module.Searcher = Mock(return_value=context)
                with patch.dict("sys.modules", {"src.searcher": module}), \
                        patch("builtins.input", side_effect=[ending]), \
                        redirect_stdout(StringIO()):
                    run_search("index/")
                context.__enter__.return_value.search.assert_not_called()
                context.__exit__.assert_called_once()

    def test_nonpositive_result_limit_is_rejected(self):
        for k in (0, -1):
            with self.subTest(k=k), self.assertRaises(ValueError):
                run_search("index/", k=k)
