"""End-to-end search tests on a tiny index built with the real pipeline.

The index is built once: JSON pages -> iter_documents -> extract_tokens ->
Indexer (small flush size, so several partials) -> merge_partials. Then
Searcher answers queries against those files on disk.
"""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.corpus import iter_documents
from src.index_io import DOC_IDS_FILE
from src.indexer import Indexer
from src.merge import merge_partials
from src.parser import extract_tokens
from src.searcher import Searcher


# One page per entry. File names keep doc_ids in this order (0, 1, 2, ...).
PAGES = [
    # 0: "machine learning" twice, once inside <title> (important).
    ("https://a.edu/ml-title", "<title>Machine Learning</title><p>machine learning course</p>"),
    # 1: "machine learning" twice, no important tags. Same tf as page 0.
    ("https://a.edu/ml-plain", "<p>machine learning</p><p>machine learning</p>"),
    # 2: "machine" only.
    ("https://a.edu/machine-shop", "<p>machine shop</p>"),
    # 3: "learning" only.
    ("https://a.edu/cooking", "<p>learning to cook</p>"),
    # 4: the only page with "acm"; also has machine learning.
    ("https://a.edu/acm", "<b>ACM</b> chapter, machine learning reading group"),
    # 5: the only page with "master".
    ("https://a.edu/mse", "<p>master of software engineering program</p>"),
    # 6: "of software engineering" but no "master".
    ("https://a.edu/lopes", "<h1>Cristina Lopes</h1><p>professor of software engineering</p>"),
]

RANKING_AVAILABLE = importlib.util.find_spec("src.ranking") is not None


def build_index(data_dir: Path, index_dir: Path) -> None:
    """Write PAGES as JSON files and run the real indexing pipeline on them."""
    domain = data_dir / "a.edu"
    domain.mkdir(parents=True)
    for number, (url, html) in enumerate(PAGES):
        page = {"url": url, "content": html, "encoding": "utf-8"}
        (domain / f"{number:02d}.json").write_text(json.dumps(page), encoding="utf-8")

    indexer = Indexer(str(index_dir), flush_every=2)  # 7 pages -> 4 partials
    for doc in iter_documents(str(data_dir)):
        indexer.add_document(doc, extract_tokens(doc.html))
    indexer.flush()
    merge_partials(indexer.partial_paths, index_dir)


def urls(results) -> set[str]:
    """The URLs of a result list, ignoring order."""
    return {result.url for result in results}


class SearcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.index_dir = root / "index"
        build_index(root / "data", cls.index_dir)
        cls.searcher = Searcher(cls.index_dir)

    @classmethod
    def tearDownClass(cls):
        cls.searcher.close()
        cls._tmp.cleanup()

    # --- AND retrieval ---------------------------------------------------

    def test_and_returns_only_pages_with_every_term(self):
        """Pages with only "machine" or only "learning" are left out."""
        results, total = self.searcher.search("machine learning", k=10)
        self.assertEqual(total, 3)
        self.assertEqual(
            urls(results),
            {"https://a.edu/ml-title", "https://a.edu/ml-plain", "https://a.edu/acm"},
        )

    def test_stop_words_are_searched_not_dropped(self):
        """'of' is indexed, and 'master' rules out the Lopes page."""
        results, total = self.searcher.search("master of software engineering")
        self.assertEqual(total, 1)
        self.assertEqual(urls(results), {"https://a.edu/mse"})

        results, total = self.searcher.search("of software engineering")
        self.assertEqual(urls(results), {"https://a.edu/mse", "https://a.edu/lopes"})

    def test_total_counts_all_matches_when_k_is_smaller(self):
        results, total = self.searcher.search("machine learning", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(total, 3)

    # --- Query normalization -------------------------------------------------

    def test_case_and_word_forms_match_the_index(self):
        """Queries use the indexing tokenizer, so case and stems line up."""
        expected = self.searcher.search("acm")
        self.assertEqual(self.searcher.search("ACM"), expected)
        self.assertEqual(urls(expected[0]), {"https://a.edu/acm"})
        # "Machines learned" stems to the same terms as "machine learning".
        self.assertEqual(
            urls(self.searcher.search("Machines learned", k=10)[0]),
            urls(self.searcher.search("machine learning", k=10)[0]),
        )

    def test_repeated_terms_do_not_change_results(self):
        self.assertEqual(
            self.searcher.search("acm acm ACM"), self.searcher.search("acm")
        )

    # --- No results ------------------------------------------------------

    def test_missing_term_gives_no_results(self):
        """One unknown term is enough: AND cannot match any page."""
        self.assertEqual(self.searcher.search("quantum"), ([], 0))
        self.assertEqual(self.searcher.search("machine quantum"), ([], 0))

    def test_empty_and_punctuation_queries_give_no_results(self):
        for query in ["", "   ", "!!!", "--- ???"]:
            self.assertEqual(self.searcher.search(query), ([], 0), query)

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.searcher.search("acm", k=0)

    # --- Index access ----------------------------------------------------

    def test_results_map_doc_ids_to_their_urls(self):
        results, _ = self.searcher.search("software engineering", k=10)
        for result in results:
            self.assertEqual(result.url, PAGES[result.doc_id][0])

    def test_shortest_posting_list_is_read_first(self):
        """'acm' (1 page) is looked up before 'machin' (4 pages)."""
        reader = self.searcher.reader
        with patch.object(reader, "lookup", wraps=reader.lookup) as lookup:
            self.searcher.search("machine acm")
        looked_up = [call.args[0] for call in lookup.call_args_list]
        self.assertEqual(looked_up, ["acm", "machin"])

    def test_missing_term_skips_disk_reads(self):
        """When a term is absent, no posting list is read at all."""
        reader = self.searcher.reader
        with patch.object(reader, "lookup", wraps=reader.lookup) as lookup:
            self.searcher.search("machine quantum")
        lookup.assert_not_called()

    def test_doc_count_is_number_of_pages(self):
        self.assertEqual(self.searcher.doc_count, len(PAGES))

    # --- Ranking (needs Person B's ranking.py) ---------------------------

    @unittest.skipUnless(RANKING_AVAILABLE, "src/ranking.py not added yet")
    def test_scores_are_sorted_best_first(self):
        results, _ = self.searcher.search("machine learning", k=10)
        scores = [result.score for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[0], 0)

    @unittest.skipUnless(RANKING_AVAILABLE, "src/ranking.py not added yet")
    def test_important_tag_page_outranks_same_tf_plain_page(self):
        """Pages 0 and 1 have the same tf; page 0 has the terms in <title>."""
        results, _ = self.searcher.search("machine learning", k=10)
        order = [result.url for result in results]
        self.assertLess(
            order.index("https://a.edu/ml-title"), order.index("https://a.edu/ml-plain")
        )


class SearcherOpenTests(unittest.TestCase):
    def test_index_without_documents_is_rejected(self):
        """An empty doc_ids.tsv means no usable index, so fail when opening."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_index(root / "data", root / "index")
            (root / "index" / DOC_IDS_FILE).write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                Searcher(root / "index")


if __name__ == "__main__":
    unittest.main()
