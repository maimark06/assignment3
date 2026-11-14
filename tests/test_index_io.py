"""Check writing the final index with offsets and reading postings by seeking."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.index_io import (
    DOC_IDS_FILE,
    INDEX_FILE,
    OFFSETS_FILE,
    IndexReader,
    IndexWriter,
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


if __name__ == "__main__":
    unittest.main()
