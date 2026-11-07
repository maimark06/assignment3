"""Checks for the in-memory inverted index."""

import unittest

from src.corpus import Document
from src.indexer import Indexer
from src.posting import Posting, format_line, parse_line


class InMemoryIndexTests(unittest.TestCase):
    def test_add_document_stores_postings_without_adding_importance_to_tf(self):
        """Each token becomes one posting; tf already includes important hits."""
        indexer = Indexer("unused-output")
        page = Document(doc_id=0, url="https://example.com/a", html="<p>ignored</p>")

        indexer.add_document(page, {"cat": (2, 1), "dog": (1, 1)})

        self.assertEqual(
            indexer.postings("cat"),
            [Posting(doc_id=0, tf=2, imp=1)],
        )
        self.assertEqual(
            indexer.postings("dog"),
            [Posting(doc_id=0, tf=1, imp=1)],
        )

    def test_later_documents_append_postings_in_doc_id_order(self):
        """The same term keeps one posting per document, in insertion order."""
        indexer = Indexer("unused-output")
        first = Document(0, "https://example.com/a", "")
        second = Document(1, "https://example.com/b", "")

        indexer.add_document(first, {"cat": (2, 1)})
        indexer.add_document(second, {"cat": (1, 0), "bird": (4, 0)})

        self.assertEqual(
            indexer.postings("cat"),
            [Posting(0, 2, 1), Posting(1, 1, 0)],
        )
        self.assertEqual(indexer.postings("bird"), [Posting(1, 4, 0)])
        self.assertEqual(indexer.postings("missing"), [])

    def test_document_with_no_tokens_is_still_remembered(self):
        """A page with no words still keeps its id and URL for doc_ids.tsv."""
        indexer = Indexer("unused-output")
        empty = Document(0, "https://example.com/empty", "")
        later = Document(1, "https://example.com/later", "")

        indexer.add_document(empty, {})
        indexer.add_document(later, {"cat": (1, 0)})

        self.assertEqual(
            indexer.documents,
            [
                (0, "https://example.com/empty"),
                (1, "https://example.com/later"),
            ],
        )
        self.assertEqual(indexer.postings("cat"), [Posting(1, 1, 0)])


class PostingLineTests(unittest.TestCase):
    def test_format_line_round_trips_through_parse_line(self):
        """A term line is term, tab, then doc_id:tf:imp fields."""
        postings = [Posting(0, 2, 1), Posting(2, 3, 0)]

        line = format_line("cat", postings)

        self.assertEqual(line, "cat\t0:2:1,2:3:0\n")
        self.assertEqual(parse_line(line), ("cat", postings))

    def test_format_line_keeps_caller_posting_order(self):
        """Sorting by doc_id is the caller's job, not the formatter's."""
        line = format_line("cat", [Posting(2, 3, 0), Posting(0, 2, 1)])

        self.assertEqual(line, "cat\t2:3:0,0:2:1\n")

    def test_empty_posting_list_is_rejected(self):
        """A term with no documents does not get an index line."""
        with self.assertRaises(ValueError):
            format_line("cat", [])
        with self.assertRaises(ValueError):
            parse_line("cat\t\n")
