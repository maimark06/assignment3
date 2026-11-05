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

    def test_invalid_json_is_logged_and_skipped_without_interrupting_iteration(self):
        """A corrupt page must not prevent later valid pages from loading."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0.json").write_text(
                json.dumps({"url": "https://example.com/first", "content": "first"}),
                encoding="utf-8",
            )
            (root / "1.json").write_text('{"url":', encoding="utf-8")
            (root / "2.json").write_text(
                json.dumps({"url": "https://example.com/last", "content": "last"}),
                encoding="utf-8",
            )

            with self.assertLogs("src.corpus", level="INFO") as logs:
                documents = list(iter_documents(directory))

        self.assertEqual(
            [(doc.doc_id, doc.url, doc.html) for doc in documents],
            [
                (0, "https://example.com/first", "first"),
                (1, "https://example.com/last", "last"),
            ],
        )
        self.assertTrue(
            any(
                record.levelname == "WARNING" and "1.json" in record.getMessage()
                for record in logs.records
            )
        )
        self.assertIn("1 invalid files skipped", logs.records[-1].getMessage())

    def test_invalid_page_fields_are_skipped(self):
        """Reject malformed page records while accepting empty HTML content."""
        invalid_pages = [
            [],
            {"content": "missing URL"},
            {"url": 123, "content": "non-string URL"},
            {"url": "   ", "content": "blank URL"},
            {"url": "#section", "content": "fragment-only URL"},
            {"url": "https://example.com/missing-content"},
            {"url": "https://example.com/null-content", "content": None},
            {"url": "https://example.com/list-content", "content": []},
        ]
        for invalid_page in invalid_pages:
            with self.subTest(page=invalid_page), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "0.json").write_text(
                    json.dumps(invalid_page), encoding="utf-8"
                )
                (root / "1.json").write_text(
                    json.dumps({"url": "https://example.com/valid", "content": ""}),
                    encoding="utf-8",
                )

                with self.assertLogs("src.corpus", level="WARNING"):
                    documents = list(iter_documents(directory))

                self.assertEqual(
                    [(doc.doc_id, doc.url, doc.html) for doc in documents],
                    [(0, "https://example.com/valid", "")],
                )

    def test_unicode_survives_invalid_file_encoding(self):
        """Preserve Unicode and continue after an undecodable JSON file."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first_html = "<title>中文 café</title>"
            last_html = "<p>日本語 naïve</p>"
            (root / "0.json").write_text(
                json.dumps(
                    {
                        "url": "https://example.com/first",
                        "content": first_html,
                        "encoding": "iso-8859-1",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8-sig",
            )
            (root / "1.json").write_bytes(b'\xff\xfe{"url": "broken"}')
            (root / "2.json").write_text(
                json.dumps(
                    {"url": "https://example.com/last", "content": last_html},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertLogs("src.corpus", level="WARNING") as logs:
                documents = list(iter_documents(directory))

        self.assertEqual(
            [(doc.doc_id, doc.url, doc.html) for doc in documents],
            [
                (0, "https://example.com/first", first_html),
                (1, "https://example.com/last", last_html),
            ],
        )
        self.assertTrue(
            any("1.json" in record.getMessage() for record in logs.records)
        )
