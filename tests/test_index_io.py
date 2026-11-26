"""Check writing the final index with offsets and reading postings by seeking."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.index_io import (
    DOC_IDS_FILE,
    DOC_LENGTHS_FILE,
    INDEX_FILE,
    OFFSETS_FILE,
    IndexReader,
    IndexWriter,
    load_doc_lengths,
    load_doc_urls,
    load_offsets,
    lookup,
)
from src.posting import Posting


# Sorted in Python string order; "café" sorts before "cat" because "f" < "t".
SAMPLE = {
    "apple": [Posting(0, 3, 1), Posting(4, 1, 0)],
    "café": [Posting(2, 1, 1)],
    "cat": [Posting(1, 2, 0), Posting(2, 5, 2), Posting(7, 1, 0)],
    "zebra": [Posting(9, 1, 0)],
}


def write_sample(index_dir: Path) -> None:
    """Write SAMPLE as a complete index (index.txt + offsets.tsv)."""
    with IndexWriter(index_dir) as writer:
        for term in sorted(SAMPLE):
            writer.write(term, SAMPLE[term])


class IndexWriterTests(unittest.TestCase):
    def test_writes_sorted_lines_and_offset_table(self):
        """index.txt has one line per term and offsets point at line starts."""
        with TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            write_sample(index_dir)

            data = (index_dir / INDEX_FILE).read_bytes()
            self.assertEqual(data.decode("utf-8").splitlines()[0], "apple\t0:3:1,4:1:0")

            offsets = load_offsets(index_dir)
            self.assertEqual(list(offsets), sorted(SAMPLE))
            for term, offset in offsets.items():
                # Each offset is a byte position where that term's line begins.
                self.assertTrue(data[offset:].startswith(f"{term}\t".encode("utf-8")))

    def test_write_line_accepts_preformatted_lines(self):
        """The merge step can pass ready-made lines instead of Posting objects."""
        with TemporaryDirectory() as tmp:
            with IndexWriter(tmp) as writer:
                writer.write_line("alpha\t1:2:0\n")
                writer.write_line("beta\t0:1:1,3:4:0\n")
                self.assertEqual(writer.term_count, 2)
            self.assertEqual(lookup("beta", tmp), [Posting(0, 1, 1), Posting(3, 4, 0)])

    def test_rejects_out_of_order_and_duplicate_terms(self):
        """Unsorted or repeated terms indicate a merge bug and must fail loudly."""
        with TemporaryDirectory() as tmp:
            writer = IndexWriter(tmp)
            writer.write("beta", [Posting(0, 1, 0)])
            with self.assertRaises(ValueError):
                writer.write("alpha", [Posting(1, 1, 0)])
            with self.assertRaises(ValueError):
                writer.write("beta", [Posting(2, 1, 0)])
            writer.abort()

    def test_rejects_malformed_lines(self):
        """Lines need a term, a tab, postings and exactly one trailing newline."""
        with TemporaryDirectory() as tmp:
            writer = IndexWriter(tmp)
            for bad in ["alpha\t1:1:0", "alpha 1:1:0\n", "\t1:1:0\n", "a\t1:1:0\nb\t2:1:0\n"]:
                with self.assertRaises(ValueError, msg=bad):
                    writer.write_line(bad)
            writer.abort()

    def test_no_offset_table_when_build_fails(self):
        """A crash inside the with-block must not leave a complete-looking index."""
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                with IndexWriter(tmp) as writer:
                    writer.write("alpha", [Posting(0, 1, 0)])
                    raise RuntimeError("merge failed")
            self.assertFalse((Path(tmp) / OFFSETS_FILE).exists())


class IndexReaderTests(unittest.TestCase):
    def test_lookup_first_middle_last_and_non_ascii_terms(self):
        """Every stored term returns exactly its own postings."""
        with TemporaryDirectory() as tmp:
            write_sample(Path(tmp))
            with IndexReader(tmp) as reader:
                self.assertEqual(len(reader), len(SAMPLE))
                for term, expected in SAMPLE.items():
                    self.assertEqual(reader.lookup(term), expected, term)
                # Repeat in reverse to make sure seeking backwards also works.
                for term in reversed(list(SAMPLE)):
                    self.assertEqual(reader.lookup(term), SAMPLE[term], term)

    def test_missing_term_returns_empty_list(self):
        with TemporaryDirectory() as tmp:
            write_sample(Path(tmp))
            with IndexReader(tmp) as reader:
                self.assertEqual(reader.lookup("missing"), [])
                self.assertNotIn("missing", reader)
                self.assertIn("cat", reader)

    def test_document_frequency_is_posting_list_length(self):
        with TemporaryDirectory() as tmp:
            write_sample(Path(tmp))
            with IndexReader(tmp) as reader:
                self.assertEqual(reader.document_frequency("cat"), 3)
                self.assertEqual(reader.document_frequency("missing"), 0)

    def test_stale_offset_table_is_detected(self):
        """Offsets from a different build must raise instead of returning wrong postings."""
        with TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            write_sample(index_dir)
            # Point "cat" at the line that actually holds "apple".
            (index_dir / OFFSETS_FILE).write_text("cat\t0\n", encoding="utf-8")
            with IndexReader(index_dir) as reader:
                with self.assertRaises(ValueError):
                    reader.lookup("cat")


class SparseOffsetTests(unittest.TestCase):
    """M3: the reader keeps only every Nth term of offsets.tsv in memory."""

    # 250 terms: "t000", "t002", ... (even numbers only), so odd numbers like
    # "t001" sort between two stored terms but are not in the index.
    TERMS = [f"t{number:03d}" for number in range(0, 500, 2)]

    def write_many(self, index_dir: Path) -> None:
        with IndexWriter(index_dir) as writer:
            for number, term in enumerate(self.TERMS):
                # Different list lengths, so posting sizes differ per term.
                writer.write(term, [Posting(d, 1, 0) for d in range(number % 7 + 1)])

    def full_sizes(self, index_dir: Path) -> dict[str, int]:
        """Expected line length of every term, from the full offset table."""
        offsets = load_offsets(index_dir)
        starts = sorted(offsets.values()) + [(index_dir / INDEX_FILE).stat().st_size]
        return {term: starts[starts.index(o) + 1] - o for term, o in offsets.items()}

    def test_every_block_size_gives_the_same_answers(self):
        """Terms at block starts, block ends and the last row all work."""
        with TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            self.write_many(index_dir)
            sizes = self.full_sizes(index_dir)
            for every in [1, 2, 3, 7, 100, 249, 250, 1000]:
                with IndexReader(index_dir, every=every) as reader:
                    self.assertEqual(len(reader), len(self.TERMS), every)
                    for number, term in enumerate(self.TERMS):
                        self.assertIn(term, reader)
                        self.assertEqual(reader.posting_size(term), sizes[term])
                        self.assertEqual(len(reader.lookup(term)), number % 7 + 1)

    def test_absent_terms_before_between_and_after(self):
        with TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            self.write_many(index_dir)
            absent = ["", "a", "t001", "t099", "t100x", "t497", "t499", "zzz"]
            for every in [1, 3, 100]:
                with IndexReader(index_dir, every=every) as reader:
                    for term in absent:
                        self.assertNotIn(term, reader)
                        self.assertEqual(reader.lookup(term), [], (every, term))
                        self.assertEqual(reader.posting_size(term), 0)

    def test_only_one_term_per_block_is_kept_in_memory(self):
        with TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            self.write_many(index_dir)
            with IndexReader(index_dir, every=100) as reader:
                self.assertEqual(reader._block_terms, ["t000", "t200", "t400"])
                self.assertFalse(hasattr(reader, "offsets"))

    def test_non_ascii_terms_use_byte_positions(self):
        """Seeking needs byte positions, which differ from character counts."""
        with TemporaryDirectory() as tmp:
            write_sample(Path(tmp))
            for every in [1, 2, 3]:
                with IndexReader(tmp, every=every) as reader:
                    for term, expected in SAMPLE.items():
                        self.assertEqual(reader.lookup(term), expected, (every, term))

    def test_unsorted_or_broken_offset_table_is_rejected(self):
        bad_tables = ["b\t0\na\t10\n", "a\t0\na\t10\n", "a 0\n", "a\tx\n"]
        for text in bad_tables:
            with TemporaryDirectory() as tmp:
                write_sample(Path(tmp))
                (Path(tmp) / OFFSETS_FILE).write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError, msg=text):
                    IndexReader(tmp)

    def test_empty_index_has_no_terms(self):
        with TemporaryDirectory() as tmp:
            with IndexWriter(tmp):
                pass
            with IndexReader(tmp) as reader:
                self.assertEqual(len(reader), 0)
                self.assertNotIn("a", reader)
                self.assertEqual(reader.lookup("a"), [])

    def test_every_must_be_positive(self):
        with TemporaryDirectory() as tmp:
            write_sample(Path(tmp))
            with self.assertRaises(ValueError):
                IndexReader(tmp, every=0)


class SearchSupportTests(unittest.TestCase):
    """Milestone 2 additions: posting_size, doc_count, load_doc_urls."""

    def test_posting_size_is_byte_length_of_each_line(self):
        """First, middle, last (ends at end of file) and non-ASCII terms."""
        with TemporaryDirectory() as tmp:
            write_sample(Path(tmp))
            lines = (Path(tmp) / INDEX_FILE).read_bytes().splitlines(keepends=True)
            expected = {line.split(b"\t")[0].decode("utf-8"): len(line) for line in lines}
            with IndexReader(tmp) as reader:
                for term in SAMPLE:
                    self.assertEqual(reader.posting_size(term), expected[term], term)
                # Sizes must add up to the whole file, with no gaps or overlaps.
                total = sum(reader.posting_size(term) for term in SAMPLE)
                self.assertEqual(total, (Path(tmp) / INDEX_FILE).stat().st_size)

    def test_posting_size_orders_longer_lists_first_and_missing_is_zero(self):
        with TemporaryDirectory() as tmp:
            write_sample(Path(tmp))
            with IndexReader(tmp) as reader:
                # "cat" has 3 postings, "zebra" has 1.
                self.assertGreater(reader.posting_size("cat"), reader.posting_size("zebra"))
                self.assertEqual(reader.posting_size("missing"), 0)

    def test_doc_count_reads_doc_ids_and_is_zero_without_it(self):
        with TemporaryDirectory() as tmp:
            write_sample(Path(tmp))
            with IndexReader(tmp) as reader:
                self.assertEqual(reader.doc_count, 0)
            (Path(tmp) / DOC_IDS_FILE).write_text(
                "0\thttps://a.com\n1\thttps://b.com\n\n2\thttps://c.com\n",
                encoding="utf-8",
            )
            with IndexReader(tmp) as reader:
                self.assertEqual(reader.doc_count, 3)

    def test_load_doc_urls_puts_each_url_at_its_doc_id(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / DOC_IDS_FILE).write_text(
                "0\thttps://a.com/x?q=1\n1\thttps://b.com\n", encoding="utf-8"
            )
            self.assertEqual(load_doc_urls(tmp), ["https://a.com/x?q=1", "https://b.com"])

    def test_load_doc_urls_rejects_gaps_and_bad_lines(self):
        """A gap would make list position differ from doc_id, so it must fail."""
        for text in ["0\thttps://a.com\n2\thttps://c.com\n", "0 https://a.com\n"]:
            with TemporaryDirectory() as tmp:
                (Path(tmp) / DOC_IDS_FILE).write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError, msg=text):
                    load_doc_urls(tmp)

    def test_load_doc_lengths_puts_each_length_at_its_doc_id(self):
        """Blank lines are skipped and a page with no tokens has length 0."""
        with TemporaryDirectory() as tmp:
            (Path(tmp) / DOC_LENGTHS_FILE).write_text(
                "0\t120\n1\t0\n\n2\t7\n", encoding="utf-8"
            )
            self.assertEqual(load_doc_lengths(tmp), [120, 0, 7])

    def test_load_doc_lengths_rejects_gaps_and_bad_lines(self):
        bad_files = [
            "0\t10\n2\t30\n",  # gap: doc_id 1 missing
            "1\t10\n",  # does not start at 0
            "0 10\n",  # no tab
            "0\t-5\n",  # negative length
            "0\tten\n",  # not a number
            "0\t\n",  # empty length
        ]
        for text in bad_files:
            with TemporaryDirectory() as tmp:
                (Path(tmp) / DOC_LENGTHS_FILE).write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError, msg=text):
                    load_doc_lengths(tmp)

    def test_load_doc_lengths_needs_the_file(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_doc_lengths(tmp)


if __name__ == "__main__":
    unittest.main()
