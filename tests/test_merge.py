"""Checks that partial indexes merge into one final index."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.corpus import Document
from src.index_io import INDEX_FILE, IndexReader
from src.indexer import Indexer
from src.merge import merge_partials
from src.posting import Posting


PAGES = [
    (0, "https://example.com/a", {"dog": (1, 0), "cat": (2, 1)}),
    (1, "https://example.com/b", {"cat": (1, 0)}),
    (2, "https://example.com/c", {"bird": (4, 0)}),
    (3, "https://example.com/d", {"cat": (3, 1)}),
]

EXPECTED_INDEX = "bird\t2:4:0\ncat\t0:2:1,1:1:0,3:3:1\ndog\t0:1:0\n"


def index_pages(out_dir: Path, flush_every: int) -> Indexer:
    """Index PAGES, then flush the leftover batch."""
    indexer = Indexer(str(out_dir), flush_every=flush_every)
    for doc_id, url, tokens in PAGES:
        indexer.add_document(Document(doc_id, url, ""), tokens)
    indexer.flush()
    return indexer


class MergePartialTests(unittest.TestCase):
    def test_several_partials_match_one_single_pass_index(self):
        """Merging every batch equals indexing the same pages in one batch."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            split = index_pages(root / "split", flush_every=2)
            whole = index_pages(root / "whole", flush_every=100)

            self.assertEqual(split.flush_count, 2)
            self.assertEqual(whole.flush_count, 1)

            merge_partials(split.partial_paths, root / "split")
            merge_partials(whole.partial_paths, root / "whole")

            merged = (root / "split" / INDEX_FILE).read_text(encoding="utf-8")
            self.assertEqual(merged, EXPECTED_INDEX)
            self.assertEqual(
                merged,
                (root / "whole" / INDEX_FILE).read_text(encoding="utf-8"),
            )
            with IndexReader(root / "split") as reader:
                self.assertEqual(
                    reader.lookup("cat"),
                    [Posting(0, 2, 1), Posting(1, 1, 0), Posting(3, 3, 1)],
                )
                self.assertEqual(reader.lookup("missing"), [])
