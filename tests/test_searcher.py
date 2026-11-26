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
from src.index_io import DOC_IDS_FILE, DOC_LENGTHS_FILE, load_doc_lengths
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


# Pages for length damping: same tf and imp for the query terms, very
# different page length. Kept apart from PAGES so the counts above stay fixed.
LENGTH_PAGES = [
    # 0: long page (the terms plus 200 filler words). Smaller doc_id, so it
    #    would win a tie; it must lose because it is longer.
    ("https://b.edu/long", "<p>machine learning " + "filler " * 200 + "</p>"),
    # 1: short page with the same counts for "machine" and "learning".
    ("https://b.edu/short", "<p>machine learning notes</p>"),
    # 2: neither term, so idf is not zero.
    ("https://b.edu/other", "<p>unrelated page</p>"),
]


def build_index(data_dir: Path, index_dir: Path, pages=PAGES) -> None:
    """Write pages as JSON files and run the real indexing pipeline on them."""
    domain = data_dir / "a.edu"
    domain.mkdir(parents=True)
    for number, (url, html) in enumerate(pages):
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


def full_matches(results) -> set[str]:
    """URLs of the results that contain every query term (AND matches)."""
    most = max(result.matched_terms for result in results)
    return {result.url for result in results if result.matched_terms == most}


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

    def test_and_matches_come_first_and_are_counted(self):
        """Only pages with both terms count; they fill the first slots."""
        results, total = self.searcher.search("machine learning", k=3)
        self.assertEqual(total, 3)
        self.assertEqual(
            urls(results),
            {"https://a.edu/ml-title", "https://a.edu/ml-plain", "https://a.edu/acm"},
        )
        self.assertEqual([result.matched_terms for result in results], [2, 2, 2])

    def test_stop_words_are_searched_not_dropped(self):
        """'of' is indexed, and 'master' rules out the Lopes page."""
        results, total = self.searcher.search("master of software engineering")
        self.assertEqual(total, 1)
        self.assertEqual(results[0].url, "https://a.edu/mse")
        self.assertEqual(full_matches(results), {"https://a.edu/mse"})

        results, total = self.searcher.search("of software engineering")
        self.assertEqual(total, 2)
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

    def test_query_with_no_known_term_gives_no_results(self):
        self.assertEqual(self.searcher.search("quantum"), ([], 0))
        self.assertEqual(self.searcher.search("quantum zebra"), ([], 0))

    def test_missing_term_falls_back_to_partial_matches(self):
        """No page has 'quantum', so AND finds 0, but 'machine' pages fill in."""
        results, total = self.searcher.search("machine quantum", k=10)
        self.assertEqual(total, 0)
        self.assertEqual(
            urls(results),
            {
                "https://a.edu/ml-title",
                "https://a.edu/ml-plain",
                "https://a.edu/machine-shop",
                "https://a.edu/acm",
            },
        )
        self.assertTrue(all(result.matched_terms == 1 for result in results))

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

    def test_missing_terms_are_never_read_from_disk(self):
        """A term not in the index is skipped before any posting list is read."""
        reader = self.searcher.reader
        with patch.object(reader, "lookup", wraps=reader.lookup) as lookup:
            self.searcher.search("machine quantum")
        self.assertEqual([call.args[0] for call in lookup.call_args_list], ["machin"])

        with patch.object(reader, "lookup", wraps=reader.lookup) as lookup:
            self.searcher.search("quantum zebra")
        lookup.assert_not_called()

    # --- Fill-in from partial matches -------------------------------------

    def test_partial_matches_fill_slots_after_and_matches(self):
        """3 AND matches, then the 2 pages that have only one of the terms."""
        results, total = self.searcher.search("machine learning", k=10)
        self.assertEqual(total, 3)
        self.assertEqual(len(results), 5)
        self.assertEqual(
            urls(results[:3]),
            {"https://a.edu/ml-title", "https://a.edu/ml-plain", "https://a.edu/acm"},
        )
        self.assertEqual(
            urls(results[3:]), {"https://a.edu/machine-shop", "https://a.edu/cooking"}
        )
        self.assertEqual([r.matched_terms for r in results], [2, 2, 2, 1, 1])

    def test_pages_with_more_query_terms_come_first(self):
        """'machine acm learning': acm page has 3 terms, ml pages 2, others 1."""
        results, total = self.searcher.search("machine acm learning", k=10)
        self.assertEqual(total, 1)
        self.assertEqual(results[0].url, "https://a.edu/acm")
        counts = [result.matched_terms for result in results]
        self.assertEqual(counts, [3, 2, 2, 1, 1])
        self.assertEqual(
            urls(results[1:3]), {"https://a.edu/ml-title", "https://a.edu/ml-plain"}
        )

    def test_partial_matches_are_ranked_by_score_within_a_level(self):
        """Same number of terms: higher score first (title page beats plain)."""
        results, _ = self.searcher.search("machine acm learning", k=10)
        self.assertEqual(
            [results[1].url, results[2].url],
            ["https://a.edu/ml-title", "https://a.edu/ml-plain"],
        )
        self.assertGreaterEqual(results[1].score, results[2].score)

    def test_fill_in_never_repeats_a_page(self):
        for query in ["machine learning", "master of software engineering", "of"]:
            results, _ = self.searcher.search(query, k=10)
            doc_ids = [result.doc_id for result in results]
            self.assertEqual(len(doc_ids), len(set(doc_ids)), query)

    def test_no_fill_in_work_when_and_fills_k(self):
        with patch.object(
            self.searcher, "_partial_results", wraps=self.searcher._partial_results
        ) as partial:
            results, _ = self.searcher.search("machine learning", k=3)
        partial.assert_not_called()
        self.assertEqual(len(results), 3)

    def test_fill_in_stops_at_k(self):
        results, total = self.searcher.search("machine learning", k=4)
        self.assertEqual(total, 3)
        self.assertEqual(len(results), 4)
        self.assertEqual(results[3].matched_terms, 1)

    def test_doc_count_is_number_of_pages(self):
        self.assertEqual(self.searcher.doc_count, len(PAGES))

    # --- Ranking (needs Person B's ranking.py) ---------------------------

    @unittest.skipUnless(RANKING_AVAILABLE, "src/ranking.py not added yet")
    def test_scores_are_sorted_best_first_within_each_level(self):
        """Score order holds among pages with the same number of query terms."""
        results, _ = self.searcher.search("machine learning", k=10)
        for count in {result.matched_terms for result in results}:
            scores = [r.score for r in results if r.matched_terms == count]
            self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(results[0].score, 0)

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

    def test_lengths_must_cover_every_document(self):
        """An old or broken doc_lengths.tsv would give pages the wrong length."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_index(root / "data", root / "index")
            (root / "index" / DOC_LENGTHS_FILE).write_text("0\t5\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                Searcher(root / "index")

    def test_missing_lengths_file_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_index(root / "data", root / "index")
            (root / "index" / DOC_LENGTHS_FILE).unlink()
            with self.assertRaises(FileNotFoundError):
                Searcher(root / "index")


class LengthDampingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.index_dir = root / "index"
        build_index(root / "data", cls.index_dir, pages=LENGTH_PAGES)
        cls.searcher = Searcher(cls.index_dir)

    @classmethod
    def tearDownClass(cls):
        cls.searcher.close()
        cls._tmp.cleanup()

    def test_lengths_and_average_come_from_doc_lengths_file(self):
        lengths = load_doc_lengths(self.index_dir)
        self.assertEqual(self.searcher.lengths, lengths)
        self.assertGreater(lengths[0], lengths[1])  # long page really is longer
        self.assertAlmostEqual(self.searcher.avg_length, sum(lengths) / len(lengths))

    def test_short_page_outranks_long_page_with_same_tf(self):
        results, total = self.searcher.search("machine learning", k=10)
        self.assertEqual(total, 2)
        self.assertEqual(
            [result.url for result in results],
            ["https://b.edu/short", "https://b.edu/long"],
        )
        self.assertGreater(results[0].score, results[1].score)


if __name__ == "__main__":
    unittest.main()
