"""Checks for the in-memory inverted index and its partial-file flush."""

from pathlib import Path
from tempfile import TemporaryDirectory
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


class PartialFlushTests(unittest.TestCase):
    def test_full_batches_become_three_sorted_partials(self):
        """Six documents with a batch size of two produce three partial files."""
        with TemporaryDirectory() as directory:
            indexer = Indexer(directory, flush_every=2)
            pages = [
                (0, "https://example.com/a", {"dog": (1, 0), "cat": (2, 1)}),
                (1, "https://example.com/b", {"cat": (1, 0)}),
                (2, "https://example.com/c", {"bird": (4, 0)}),
                (3, "https://example.com/d", {"bird": (1, 1)}),
                (5, "https://example.com/f", {"ant": (1, 0)}),
                (4, "https://example.com/e", {"ant": (2, 0)}),
            ]
            for doc_id, url, tokens in pages:
                indexer.add_document(Document(doc_id, url, ""), tokens)

            self.assertEqual(indexer.flush_count, 3)
            self.assertEqual(indexer.postings("cat"), [])
            self.assertEqual(indexer.postings("ant"), [])
            partials = indexer.partial_paths
            self.assertEqual(
                [path.name for path in partials],
                ["partial-000.txt", "partial-001.txt", "partial-002.txt"],
            )
            self.assertEqual(
                partials[0].read_text(encoding="utf-8"),
                "cat\t0:2:1,1:1:0\ndog\t0:1:0\n",
            )
            self.assertEqual(
                partials[2].read_text(encoding="utf-8"),
                "ant\t4:2:0,5:1:0\n",
            )
            self.assertEqual(
                (Path(directory) / "doc_ids.tsv").read_text(encoding="utf-8"),
                "0\thttps://example.com/a\n"
                "1\thttps://example.com/b\n"
                "2\thttps://example.com/c\n"
                "3\thttps://example.com/d\n"
                "5\thttps://example.com/f\n"
                "4\thttps://example.com/e\n",
            )

    def test_short_final_batch_is_written_by_an_explicit_flush(self):
        """Documents past the last full batch stay in memory until flush."""
        with TemporaryDirectory() as directory:
            indexer = Indexer(directory, flush_every=3)
            indexer.add_document(Document(0, "https://example.com/a", ""), {"cat": (1, 0)})
            indexer.add_document(Document(1, "https://example.com/b", ""), {"cat": (1, 1)})
            indexer.add_document(Document(2, "https://example.com/empty", ""), {})
            indexer.add_document(Document(3, "https://example.com/c", ""), {"dog": (2, 1)})

            self.assertEqual(indexer.flush_count, 1)
            self.assertEqual(indexer.postings("dog"), [Posting(3, 2, 1)])
            self.assertFalse((Path(directory) / "partials" / "partial-001.txt").exists())

            indexer.flush()

            self.assertEqual(indexer.flush_count, 2)
            self.assertEqual(indexer.postings("dog"), [])
            self.assertEqual(
                (Path(directory) / "partials" / "partial-001.txt").read_text(encoding="utf-8"),
                "dog\t3:2:1\n",
            )
            self.assertEqual(
                (Path(directory) / "doc_ids.tsv").read_text(encoding="utf-8"),
                "0\thttps://example.com/a\n"
                "1\thttps://example.com/b\n"
                "2\thttps://example.com/empty\n"
                "3\thttps://example.com/c\n",
            )

    def test_flush_with_no_postings_creates_no_partial_file(self):
        """An empty in-memory map must not leave a blank partial behind."""
        with TemporaryDirectory() as directory:
            indexer = Indexer(directory, flush_every=2)

            indexer.flush()

            self.assertEqual(indexer.flush_count, 0)
            self.assertEqual(indexer.partial_paths, [])
            self.assertFalse((Path(directory) / "partials").exists())
