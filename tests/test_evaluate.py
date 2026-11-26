"""Checks for the evaluation helper (scripts/evaluate.py) on a tiny index."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.evaluate import DEFAULT_QUERIES, evaluate, main, read_queries
from tests.test_searcher import build_index


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.index_dir = cls.root / "index"
        build_index(cls.root / "data", cls.index_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_read_queries_skips_blanks_and_comments(self):
        path = self.root / "q.txt"
        path.write_text("# header\n\n machine learning \n# note\nACM\n", encoding="utf-8")
        self.assertEqual(read_queries(path), ["machine learning", "ACM"])

    def test_query_file_without_queries_is_rejected(self):
        path = self.root / "empty.txt"
        path.write_text("# only a comment\n\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_queries(path)

    def test_default_query_file_has_the_20_candidates(self):
        queries = read_queries(DEFAULT_QUERIES)
        self.assertEqual(len(queries), 20)
        self.assertEqual(len(set(queries)), 20)

    def test_report_lists_matches_results_and_partial_marks(self):
        report = evaluate(self.index_dir, ["machine learning", "quantum"], 5, 2, "test")
        self.assertIn("Evaluation run: test", report)
        self.assertIn("7 documents", report)
        self.assertIn("== 01. machine learning", report)
        self.assertIn("Matches: 3 |", report)
        # 3 AND matches, then 2 pages with one of the two terms.
        self.assertEqual(report.count("partial 1/2 terms"), 2)
        self.assertIn("== 02. quantum", report)
        self.assertIn("(no results)", report)
        self.assertIn("Summary", report)

    def test_out_file_is_written_and_never_overwritten(self):
        queries = self.root / "one.txt"
        queries.write_text("acm\n", encoding="utf-8")
        out = self.root / "eval" / "baseline.txt"
        args = ["--index", str(self.index_dir), "--queries", str(queries), "--out", str(out)]

        with redirect_stdout(StringIO()):
            main(args)
        saved = out.read_text(encoding="utf-8")
        self.assertIn("https://a.edu/acm", saved)

        with self.assertRaises(SystemExit), redirect_stdout(StringIO()):
            main(args)
        self.assertEqual(out.read_text(encoding="utf-8"), saved)


if __name__ == "__main__":
    unittest.main()
