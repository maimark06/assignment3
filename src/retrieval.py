"""Boolean AND over doc_id-sorted posting lists.

The indexer and the merge write every posting list in increasing doc_id
order. Search relies on that order: two pointers can walk each list once
and keep only documents that appear in every list.
"""

from .posting import Posting


def intersect(posting_lists: list[list[Posting]]) -> dict[int, list[Posting]]:
    """Return documents present in every list.

    Each input list is sorted by doc_id. The value for a document is one
    Posting from each input list, in the same order as posting_lists, so
    ranking can pair matches[doc_id][i] with the document frequency of term i.

    An empty input, or any empty list, matches nothing. Lists are merged
    from first to last, and an empty running result stops the merge.
    """
    if not posting_lists or any(len(postings) == 0 for postings in posting_lists):
        return {}

    # The searcher passes the shortest list first, so the running AND starts small.
    current: list[tuple[int, list[Posting]]] = [
        (posting.doc_id, [posting]) for posting in posting_lists[0]
    ]

    for nxt in posting_lists[1:]:
        current = _intersect_two(current, nxt)
        if not current:
            return {}

    return {doc_id: postings for doc_id, postings in current}


def _intersect_two(
    left: list[tuple[int, list[Posting]]],
    right: list[Posting],
) -> list[tuple[int, list[Posting]]]:
    """Merge a running AND result with the next sorted posting list.

    Both sides increase by doc_id, so each pointer only moves forward.
    Equal ids are a hit. A smaller id on one side is missing from the other.
    """
    i = 0
    j = 0
    merged: list[tuple[int, list[Posting]]] = []
    while i < len(left) and j < len(right):
        left_id = left[i][0]
        right_id = right[j].doc_id
        if left_id == right_id:
            merged.append((left_id, left[i][1] + [right[j]]))
            i += 1
            j += 1
        elif left_id < right_id:
            i += 1
        else:
            j += 1
    return merged
