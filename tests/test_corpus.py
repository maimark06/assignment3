"""Regression checks for loading crawled documents."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.corpus import iter_documents


class IterDocumentsTests(unittest.TestCase):
    def test_fragment_duplicates_keep_first_page_and_consecutive_ids(self):
        """Fragment variants share a document, without leaving gaps in IDs."""
        pages = [
            {"url": "https://example.com/page#first", "content": "first page"},
            {"url": "https://example.com/page#second", "content": "duplicate"},
            {"url": "https://example.com/other#section", "content": "other page"},
        ]

        with TemporaryDirectory() as directory:
            domain = Path(directory) / "example.com"
            domain.mkdir()
            for number, page in enumerate(pages):
                (domain / f"{number}.json").write_text(
                    json.dumps(page), encoding="utf-8"
                )

            documents = list(iter_documents(directory))

        self.assertEqual(
            [(doc.doc_id, doc.url, doc.html) for doc in documents],
            [
                (0, "https://example.com/page", "first page"),
                (1, "https://example.com/other", "other page"),
            ],
        )
