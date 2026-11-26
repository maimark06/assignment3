"""Checks for the in-memory inverted index and its partial-file flush."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.corpus import MAX_FILE_BYTES, Document
from src.index_io import IndexReader, load_doc_lengths, load_doc_urls
from src.indexer import Indexer
from src.main import run_build
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


class DocumentLengthTests(unittest.TestCase):
    def test_flush_writes_raw_token_counts_beside_doc_ids(self):
        """doc_lengths.tsv is doc_id, tab, total stemmed occurrences.

        Important-tag hits are already inside tf, so they are not added again.
        A page with no tokens is stored as 0.
        """
        with TemporaryDirectory() as directory:
            indexer = Indexer(directory, flush_every=10)
            indexer.add_document(
                Document(0, "https://example.com/a", ""),
                {"cat": (2, 1), "dog": (1, 1)},
            )
            indexer.add_document(
                Document(1, "https://example.com/empty", ""),
                {},
            )
            indexer.flush()

            lengths = Path(directory) / "doc_lengths.tsv"
            self.assertTrue(lengths.is_file())
            self.assertEqual(
                lengths.read_text(encoding="utf-8"),
                "0\t3\n1\t0\n",
            )

    def test_later_flush_rewrites_lengths_for_every_accepted_document(self):
        """Lengths accumulate like doc_ids.tsv: one line per document so far."""
        with TemporaryDirectory() as directory:
            indexer = Indexer(directory, flush_every=1)
            indexer.add_document(
                Document(0, "https://example.com/a", ""),
                {"cat": (4, 0)},
            )
            indexer.add_document(
                Document(2, "https://example.com/b", ""),
                {"dog": (1, 2), "bird": (2, 0)},
            )

            self.assertEqual(
                (Path(directory) / "doc_lengths.tsv").read_text(encoding="utf-8"),
                "0\t4\n2\t3\n",
            )


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

    def test_posting_budget_flushes_before_the_document_threshold(self):
        """Memory is bounded by how many Posting objects are in the batch."""
        with TemporaryDirectory() as directory:
            indexer = Indexer(directory, flush_every=100, flush_postings=3)
            indexer.add_document(
                Document(0, "https://example.com/a", ""),
                {"cat": (1, 0), "dog": (1, 0)},
            )
            self.assertEqual(indexer.flush_count, 0)

            indexer.add_document(
                Document(1, "https://example.com/b", ""),
                {"bird": (2, 1), "ant": (1, 0)},
            )

            self.assertEqual(indexer.flush_count, 1)
            self.assertEqual(indexer.postings("bird"), [])
            self.assertEqual(
                (Path(directory) / "partials" / "partial-000.txt").read_text(
                    encoding="utf-8"
                ),
                "ant\t1:1:0\nbird\t1:2:1\ncat\t0:1:0\ndog\t0:1:0\n",
            )

    def test_one_document_over_the_posting_budget_is_stored_then_flushed(self):
        """A single wide page is not dropped just because it exceeds the budget."""
        with TemporaryDirectory() as directory:
            indexer = Indexer(directory, flush_every=100, flush_postings=2)
            indexer.add_document(
                Document(0, "https://example.com/wide", ""),
                {"ant": (1, 0), "bird": (1, 0), "cat": (1, 0)},
            )

            self.assertEqual(indexer.flush_count, 1)
            self.assertEqual(indexer.postings("cat"), [])
            self.assertEqual(
                (Path(directory) / "partials" / "partial-000.txt").read_text(
                    encoding="utf-8"
                ),
                "ant\t0:1:0\nbird\t0:1:0\ncat\t0:1:0\n",
            )

    def test_empty_page_does_not_spend_the_posting_budget(self):
        """A page with no tokens adds a length of 0 and no Posting objects."""
        with TemporaryDirectory() as directory:
            indexer = Indexer(directory, flush_every=100, flush_postings=1)
            indexer.add_document(Document(0, "https://example.com/empty", ""), {})

            self.assertEqual(indexer.flush_count, 0)
            self.assertEqual(indexer.postings("cat"), [])

    def test_posting_budget_must_be_at_least_one(self):
        with self.assertRaises(ValueError):
            Indexer("unused-output", flush_postings=0)


class OversizedFileBuildTests(unittest.TestCase):
    def test_build_never_assigns_an_id_to_an_oversized_file(self):
        """A file over the byte limit is absent, and later pages keep consecutive ids."""
        pages = [
            {"url": "https://example.com/a", "content": "<p>small cat</p>"},
            {"url": "https://example.com/b", "content": "giant " * (MAX_FILE_BYTES // 6 + 1)},
            {"url": "https://example.com/c", "content": "<p>small dog</p>"},
        ]
        with TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            out = Path(directory) / "index"
            data.mkdir()
            for number, page in enumerate(pages):
                (data / f"{number}.json").write_text(json.dumps(page), encoding="utf-8")

            with self.assertLogs("src.corpus", level="WARNING"), redirect_stdout(StringIO()):
                run_build(str(data), str(out), flush_every=1)

            self.assertEqual(
                load_doc_urls(out),
                ["https://example.com/a", "https://example.com/c"],
            )
            self.assertEqual(load_doc_lengths(out), [2, 2])
            with IndexReader(out) as reader:
                self.assertNotIn("giant", reader)
                self.assertEqual([p.doc_id for p in reader.lookup("small")], [0, 1])
