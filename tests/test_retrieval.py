"""AND intersection of doc_id-sorted posting lists."""

import unittest

from src.posting import Posting
from src.retrieval import intersect


def posting(doc_id: int, tf: int = 1, imp: int = 0) -> Posting:
    """One posting. tf and imp differ so each list stays recognizable."""
    return Posting(doc_id, tf, imp)


class IntersectTests(unittest.TestCase):
    def test_no_lists_matches_nothing(self):
        self.assertEqual(intersect([]), {})

    def test_empty_list_matches_nothing(self):
        """A term with no postings, or an empty running result, ends the AND."""
        one = [posting(1, tf=2)]
        self.assertEqual(intersect([[]]), {})
        self.assertEqual(intersect([one, []]), {})
        self.assertEqual(intersect([[], one]), {})

    def test_single_list_returns_every_document(self):
        postings = [posting(1, tf=2), posting(4, tf=3, imp=1)]
        self.assertEqual(
            intersect([postings]),
            {
                1: [posting(1, tf=2)],
                4: [posting(4, tf=3, imp=1)],
            },
        )

    def test_disjoint_lists_match_nothing(self):
        left = [posting(1, tf=2), posting(2, tf=3)]
        right = [posting(3, tf=4), posting(4, tf=5)]
        self.assertEqual(intersect([left, right]), {})

    def test_partial_overlap_keeps_shared_documents_only(self):
        """Docs present on only one side are dropped, from either pointer."""
        left = [posting(2, tf=1), posting(5, tf=3, imp=1), posting(9, tf=4)]
        right = [posting(1, tf=7), posting(2, tf=2), posting(5, tf=1), posting(8, tf=6)]
        self.assertEqual(
            intersect([left, right]),
            {
                2: [posting(2, tf=1), posting(2, tf=2)],
                5: [posting(5, tf=3, imp=1), posting(5, tf=1)],
            },
        )

    def test_same_documents_keep_one_posting_per_list_in_input_order(self):
        first = [posting(2, tf=1), posting(5, tf=3, imp=1)]
        second = [posting(2, tf=2), posting(5, tf=1)]
        self.assertEqual(
            intersect([first, second]),
            {
                2: [posting(2, tf=1), posting(2, tf=2)],
                5: [posting(5, tf=3, imp=1), posting(5, tf=1)],
            },
        )

    def test_three_lists_keep_only_the_common_document(self):
        """A doc missing from any one list is not a match."""
        first = [posting(2, tf=1), posting(5, tf=1), posting(9, tf=1)]
        second = [posting(2, tf=2), posting(5, tf=2), posting(8, tf=2)]
        third = [posting(5, tf=3, imp=1), posting(9, tf=3)]
        self.assertEqual(
            intersect([first, second, third]),
            {5: [posting(5, tf=1), posting(5, tf=2), posting(5, tf=3, imp=1)]},
        )

    def test_posting_order_follows_the_lists_as_given(self):
        """Ranking pairs matches[doc_id][i] with the df of input list i."""
        longer = [posting(1, tf=9), posting(2, tf=9)]
        shorter = [posting(2, tf=1, imp=1)]
        self.assertEqual(
            intersect([longer, shorter]),
            {2: [posting(2, tf=9), posting(2, tf=1, imp=1)]},
        )


if __name__ == "__main__":
    unittest.main()
