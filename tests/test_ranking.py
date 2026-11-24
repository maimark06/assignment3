"""Damped tf-idf ranking with the important-tag boost."""

import math
import unittest

from src.posting import Posting
from src.ranking import B, IMP_WEIGHT, K1, rank


def posting(doc_id: int, tf: int, imp: int = 0) -> Posting:
    return Posting(doc_id, tf, imp)


def lengths_for(*doc_ids: int, length: int = 10) -> list[int]:
    """Equal lengths, long enough to index every doc_id under test."""
    return [length] * (max(doc_ids) + 1)


def weight(
    tf: int,
    imp: int,
    df: int,
    n_docs: int,
    length: int = 10,
    avg_length: float = 10,
) -> float:
    """One term's score under the damped formula."""
    eff_tf = tf + IMP_WEIGHT * imp
    norm = 1 - B + B * (length / avg_length)
    tf_part = eff_tf * (K1 + 1) / (eff_tf + K1 * norm)
    return tf_part * math.log10(n_docs / df)


class RankTests(unittest.TestCase):
    def test_important_weight_starts_at_one(self):
        """An important hit is already inside tf, so weight 1 counts it twice."""
        self.assertEqual(IMP_WEIGHT, 1)

    def test_damping_constants_are_named(self):
        """K1 and B stay tunable without editing the score formula."""
        self.assertGreater(K1, 0)
        self.assertGreater(B, 0)
        self.assertLess(B, 1)

    def test_score_matches_the_damped_formula(self):
        """One plain hit and one important hit, both with df 10 in 100 docs."""
        matches = {
            7: [posting(7, tf=1, imp=0), posting(7, tf=1, imp=1)],
        }
        ranked = rank(
            matches, [10, 10], n_docs=100, k=5,
            lengths=lengths_for(7), avg_length=10,
        )
        expected = weight(1, 0, 10, 100) + weight(1, 1, 10, 100)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0][0], 7)
        self.assertAlmostEqual(ranked[0][1], expected)

    def test_important_tag_outranks_the_same_tf(self):
        matches = {
            1: [posting(1, tf=2, imp=0)],
            4: [posting(4, tf=2, imp=1)],
        }
        ranked = rank(
            matches, [10], n_docs=100, k=2,
            lengths=lengths_for(1, 4), avg_length=10,
        )
        self.assertEqual([doc_id for doc_id, _score in ranked], [4, 1])

    def test_rarer_term_contributes_more(self):
        """Same tf, length, and document; only the document frequency changes."""
        plain = {3: [posting(3, tf=1, imp=0)]}
        rare = rank(
            plain, [10], n_docs=1000, k=1,
            lengths=lengths_for(3), avg_length=10,
        )[0][1]
        common = rank(
            plain, [100], n_docs=1000, k=1,
            lengths=lengths_for(3), avg_length=10,
        )[0][1]
        self.assertGreater(rare, common)
        self.assertAlmostEqual(rare, weight(1, 0, 10, 1000))
        self.assertAlmostEqual(common, weight(1, 0, 100, 1000))

    def test_tie_goes_to_the_smaller_doc_id(self):
        matches = {
            5: [posting(5, tf=2, imp=0)],
            2: [posting(2, tf=2, imp=0)],
        }
        ranked = rank(
            matches, [10], n_docs=100, k=2,
            lengths=lengths_for(5, 2), avg_length=10,
        )
        self.assertEqual([doc_id for doc_id, _score in ranked], [2, 5])
        self.assertAlmostEqual(ranked[0][1], ranked[1][1])

    def test_top_k_keeps_the_highest_scores(self):
        matches = {
            1: [posting(1, tf=1)],
            2: [posting(2, tf=4)],
            3: [posting(3, tf=2)],
        }
        ranked = rank(
            matches, [5], n_docs=50, k=2,
            lengths=lengths_for(1, 2, 3), avg_length=10,
        )
        self.assertEqual([doc_id for doc_id, _score in ranked], [2, 3])

    def test_k_larger_than_the_match_set_returns_every_match(self):
        matches = {
            8: [posting(8, tf=1)],
            3: [posting(3, tf=3)],
        }
        ranked = rank(
            matches, [4], n_docs=40, k=10,
            lengths=lengths_for(8, 3), avg_length=10,
        )
        self.assertEqual([doc_id for doc_id, _score in ranked], [3, 8])

    def test_no_matches_returns_an_empty_list(self):
        self.assertEqual(
            rank({}, [10], n_docs=100, k=5, lengths=[], avg_length=10),
            [],
        )

    def test_same_term_frequency_scores_a_shorter_page_higher(self):
        """Length damps the score; the stored count does not include imp."""
        matches = {
            0: [posting(0, tf=3, imp=0)],
            1: [posting(1, tf=3, imp=0)],
        }
        ranked = rank(
            matches, [10], n_docs=100, k=2,
            lengths=[40, 400], avg_length=100,
        )
        self.assertEqual([doc_id for doc_id, _score in ranked], [0, 1])
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_very_long_page_with_huge_tf_loses_to_a_medium_page(self):
        """A dump that repeats a term still loses to a normal relevant page."""
        matches = {
            0: [posting(0, tf=80, imp=0)],
            1: [posting(1, tf=5, imp=0)],
        }
        ranked = rank(
            matches, [10], n_docs=1000, k=2,
            lengths=[20000, 100], avg_length=100,
        )
        self.assertEqual([doc_id for doc_id, _score in ranked], [1, 0])

    def test_tiny_page_does_not_beat_a_normal_relevant_page(self):
        """Dividing by length would rank a one-token page above a real one."""
        matches = {
            0: [posting(0, tf=1, imp=0)],
            1: [posting(1, tf=4, imp=0)],
        }
        ranked = rank(
            matches, [10], n_docs=1000, k=2,
            lengths=[1, 100], avg_length=100,
        )
        self.assertEqual([doc_id for doc_id, _score in ranked], [1, 0])


if __name__ == "__main__":
    unittest.main()
