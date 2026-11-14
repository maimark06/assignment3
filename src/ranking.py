"""Rank AND matches by tf-idf with an important-tag boost.

For query term t in document d:

    eff_tf = tf + IMP_WEIGHT * imp
    w(t, d) = (1 + log10(eff_tf)) * log10(N / df_t)
    score(d) = sum of w(t, d) over the query terms

tf already includes occurrences inside important tags, so IMP_WEIGHT of 1
counts each of those hits a second time. Cosine length normalization is
left for milestone 3.
"""

import heapq
import math

from .posting import Posting

# An important-tag occurrence is already inside tf. Adding it again makes
# that hit count twice.
IMP_WEIGHT = 1


def rank(
    matches: dict[int, list[Posting]],
    dfs: list[int],
    n_docs: int,
    k: int,
) -> list[tuple[int, float]]:
    """Return the top k (doc_id, score) pairs, highest score first.

    dfs[i] is the document frequency of the term behind matches[doc_id][i].
    A higher score wins. An equal score goes to the smaller doc_id. k larger
    than the match set returns every match.
    """
    scored = [
        (doc_id, _score(postings, dfs, n_docs))
        for doc_id, postings in matches.items()
    ]
    # (score, -doc_id): higher score first, and a smaller doc_id breaks a tie.
    return heapq.nlargest(k, scored, key=lambda item: (item[1], -item[0]))


def _score(postings: list[Posting], dfs: list[int], n_docs: int) -> float:
    """Sum the tf-idf weight of each query term in this document."""
    total = 0.0
    for posting, df in zip(postings, dfs):
        eff_tf = posting.tf + IMP_WEIGHT * posting.imp
        total += (1 + math.log10(eff_tf)) * math.log10(n_docs / df)
    return total
