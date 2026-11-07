"""Check how corpus loading and HTML parsing work together."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from src.corpus import iter_documents
from src.parser import extract_tokens


class PipelineTests(unittest.TestCase):
    def test_json_page_becomes_document_and_token_counts(self):
        """Connect real loading and parsing with controlled tokenizer output."""
        page = {
            "url": "https://example.com/page#section",
            "content": (
                "<title>Running</title><p>runs dog</p>"
                "<strong>dog</strong><script>hidden</script>"
            ),
            "encoding": "utf-8",
        }
        tokenizer = ModuleType("src.tokenizer")
        tokenizer.tokenize = Mock(
            side_effect=[["run", "run", "dog", "dog"], ["run", "dog"]]
        )

        with TemporaryDirectory() as directory:
            domain = Path(directory) / "example.com"
            domain.mkdir()
            (domain / "page.json").write_text(json.dumps(page), encoding="utf-8")

            with patch.dict("sys.modules", {"src.tokenizer": tokenizer}):
                results = [
                    (doc.doc_id, doc.url, extract_tokens(doc.html))
                    for doc in iter_documents(directory)
                ]

        self.assertEqual(
            results,
            [(0, "https://example.com/page", {"run": (2, 1), "dog": (2, 1)})],
        )
        self.assertEqual(tokenizer.tokenize.call_count, 2)
        tokenizer.tokenize.assert_any_call("Running runs dog dog")
        tokenizer.tokenize.assert_any_call("Running dog")
