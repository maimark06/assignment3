"""Checks for the in-memory inverted index."""

import unittest

from src.corpus import Document
from src.indexer import Indexer
from src.posting import Posting


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
