"""Rank AND matches by damped tf-idf with an important-tag boost.

For query term t in document d:

    eff_tf = tf + IMP_WEIGHT * imp
    norm = 1 - B + B * (length_d / avg_length)
    tf_part = eff_tf * (K1 + 1) / (eff_tf + K1 * norm)
    w(t, d) = tf_part * log10(N / df_t)
    score(d) = sum of w(t, d) over the query terms

tf already includes occurrences inside important tags, so IMP_WEIGHT of 1
counts each of those hits a second time. Document length is the raw token
count from the build, not eff_tf, so IMP_WEIGHT can be tuned later.
"""

import heapq
import math

from .posting import Posting

# An important-tag occurrence is already inside tf. Adding it again makes
# that hit count twice.
IMP_WEIGHT = 1

# BM25-style saturation (K1) and length damping (B). B below 1 keeps a
# one-token page from beating a normal page that actually uses the term.
K1 = 1.2
B = 0.75


def rank(
    matches: dict[int, list[Posting]],
    dfs: list[int],
    n_docs: int,
    k: int,
    lengths: list[int],
    avg_length: float,
) -> list[tuple[int, float]]:
    """Return the top k (doc_id, score) pairs, highest score first.

    dfs[i] is the document frequency of the term behind matches[doc_id][i].
    lengths[doc_id] is that document's raw token count. avg_length is the
    mean used in the length norm. A higher score wins. An equal score goes
    to the smaller doc_id. k larger than the match set returns every match.
    """
    scored = [
        (doc_id, _score(postings, dfs, n_docs, lengths, avg_length))
        for doc_id, postings in matches.items()
    ]
    # (score, -doc_id): higher score first, and a smaller doc_id breaks a tie.
    return heapq.nlargest(k, scored, key=lambda item: (item[1], -item[0]))


def _score(
    postings: list[Posting],
    dfs: list[int],
    n_docs: int,
    lengths: list[int],
    avg_length: float,
) -> float:
    """Sum the damped tf-idf weight of each query term in this document."""
    if not postings:
        return 0.0
    norm = 1 - B + B * (lengths[postings[0].doc_id] / avg_length)
    total = 0.0
    for posting, df in zip(postings, dfs):
        eff_tf = posting.tf + IMP_WEIGHT * posting.imp
        tf_part = eff_tf * (K1 + 1) / (eff_tf + K1 * norm)
        total += tf_part * math.log10(n_docs / df)
    return total
