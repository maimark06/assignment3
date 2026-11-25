"""Check Person A's console against the real search pipeline on temporary data."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.corpus import Document
from src.indexer import Indexer
from src.main import main
from src.merge import merge_partials
from src.parser import extract_tokens


class ConsoleIntegrationTests(unittest.TestCase):
    def test_search_command_uses_real_temporary_index(self):
        """Exercise command dispatch, parsing, disk retrieval, and console output."""
        pages = [
            ("https://example.com/acm", "<title>ACM</title><p>student chapter</p>"),
            ("https://example.com/program", "<h1>Master of Software Engineering</h1>"),
            ("https://example.com/other", "<p>unrelated page</p>"),
        ]
        with TemporaryDirectory() as directory:
            index_dir = Path(directory) / "index"
            indexer = Indexer(str(index_dir), flush_every=1)
            for doc_id, (url, html) in enumerate(pages):
                document = Document(doc_id=doc_id, url=url, html=html)
                indexer.add_document(document, extract_tokens(html))
            indexer.flush()
            merge_partials(indexer.partial_paths, index_dir)

            output = StringIO()
            with patch(
                "builtins.input",
                side_effect=[" ", "ACM acm", "master of software engineering", "!!!", "quit"],
            ), redirect_stdout(output):
                main(["search", "--index", str(index_dir), "--top", "1"])

            transcript = output.getvalue()
            self.assertIn("Search ready.", transcript)
            self.assertEqual(transcript.count("Matches: 1 |"), 2)
            self.assertIn("1. https://example.com/acm (score:", transcript)
            self.assertIn("1. https://example.com/program (score:", transcript)
            self.assertNotIn("https://example.com/other", transcript)
            self.assertIn("Matches: 0 |", transcript)
            self.assertIn("No results found.", transcript)
            self.assertEqual(transcript.count("Search time:"), 3)
            self.assertIn("Goodbye.", transcript)

        self.assertFalse(index_dir.exists())
