"""Answer one query: parse it, read posting lists from disk, AND them, rank.

Query flow (owner in brackets):

    query text
      -> parse_query        stemmed, deduplicated terms          [A, query.py]
      -> drop missing terms they cannot match any page           [C, here]
      -> posting_size       order terms shortest list first      [C, index_io]
      -> IndexReader.lookup seek() to each term's line on disk   [C, index_io]
      -> intersect          docs that contain every term         [B, retrieval.py]
      -> rank               damped tf-idf + important boost,
                            long pages damped by length, top k   [B, ranking.py]
      -> fill-in            fewer than k AND matches: add pages
                            with the most query terms            [C, here]
      -> doc_id -> URL      from doc_ids.tsv                     [C, here]

Fill-in rule: pages with every query term (AND matches) always come first.
Remaining slots go to pages with more query terms before pages with fewer;
within the same number of terms, higher score first, then smaller doc_id.
total_matches counts only the AND matches.

Memory: only the offset table, the doc_id -> URL list and the doc_id ->
length list stay in RAM.
Posting lists are read from index.txt for each query and then discarded,
so the full index is never loaded (developer-option rule).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .index_io import IndexReader, load_doc_lengths, load_doc_urls
from .posting import Posting
from .query import parse_query
from .ranking import rank
from .retrieval import intersect

PathLike = Union[str, Path]


# --- Search -----------------------------------------------------------------

@dataclass
class Result:
    """One ranked search result.

    matched_terms is how many distinct query terms the page contains. It
    equals the number of query terms for an AND match and is smaller for a
    fill-in (partial) match.
    """

    doc_id: int
    url: str
    score: float
    matched_terms: int


class Searcher:
    """Open the index once, then answer many queries.

        with Searcher("index/") as searcher:
            results, total_matches = searcher.search("machine learning", k=5)
    """

    def __init__(self, index_dir: PathLike) -> None:
        # Offset table only; posting lists stay on disk.
        self.reader = IndexReader(index_dir)
        try:
            # urls[doc_id] is that document's URL (~54k strings).
            self.urls = load_doc_urls(index_dir)
            # lengths[doc_id] is that document's token count (~54k ints).
            self.lengths = load_doc_lengths(index_dir)
        except Exception:
            self.reader.close()  # do not leave index.txt open on failure
            raise
        # N for idf. Comes from the same file as the URLs, so they agree.
        self.doc_count = len(self.urls)
        if self.doc_count == 0:
            self.reader.close()
            raise ValueError(f"No documents in {index_dir}; run the build first")
        if len(self.lengths) != self.doc_count:
            # Both files are written together by the indexer; a mismatch means
            # an old or broken build, and lengths[doc_id] would be wrong.
            self.reader.close()
            raise ValueError(
                f"{index_dir}: doc_lengths.tsv has {len(self.lengths)} documents "
                f"but doc_ids.tsv has {self.doc_count}; rebuild the index"
            )
        # Average page length over every indexed page (the usual BM25 choice).
        # Pages with no tokens count too; they never match a query, so they
        # only lower the average slightly. max(..., 1) avoids dividing by 0.
        self.avg_length = max(sum(self.lengths) / self.doc_count, 1.0)

    def search(self, query: str, k: int = 5) -> tuple[list[Result], int]:
        """Return (up to k results best first, number of AND matches).

        AND matches (pages with every query term) come first. If there are
        fewer than k, partial matches fill the rest (see the module notes).
        The count is AND matches only, so it can be 0 while partial matches
        are still returned, e.g. when one query term is not in the index.

        Returns ([], 0) for a query with no terms or with no term in the index.
        """
        if k < 1:
            raise ValueError("k must be at least 1")

        terms = parse_query(query)
        if not terms:
            return [], 0
        # A term not in the index matches no page; checking costs no disk read.
        present = [term for term in terms if term in self.reader]
        if not present:
            return [], 0

        # Shortest posting list first: intersecting from the rarest term keeps
        # the running result small. posting_size reads no postings.
        present.sort(key=self.reader.posting_size)

        posting_lists = [self.reader.lookup(term) for term in present]
        # df of each term = length of its posting list, in the same order.
        dfs = [len(postings) for postings in posting_lists]

        # A missing term means no page has every query term.
        all_present = len(present) == len(terms)
        matches = intersect(posting_lists) if all_present else {}

        results = [
            Result(doc_id, self.urls[doc_id], score, len(terms))
            for doc_id, score in self._rank(matches, dfs, k)
        ]
        if len(results) < k:
            results += self._partial_results(
                posting_lists, dfs, k - len(results), skip_size=len(terms)
            )
        return results, len(matches)

    def _rank(
        self, matches: dict[int, list[Posting]], dfs: list[int], k: int
    ) -> list[tuple[int, float]]:
        """Person B's rank with this index's N and page lengths."""
        if not matches:
            return []
        # Longer-than-average pages get a smaller tf part, so a huge dump
        # that mentions a term many times does not beat a focused page.
        return rank(
            matches,
            dfs,
            self.doc_count,
            k,
            lengths=self.lengths,
            avg_length=self.avg_length,
        )

    def _partial_results(
        self,
        posting_lists: list[list[Posting]],
        dfs: list[int],
        needed: int,
        skip_size: int,
    ) -> list[Result]:
        """Best `needed` pages that contain some, but not all, query terms.

        Pages are grouped by which terms they contain, so each group can go
        through rank with the dfs of exactly those terms. Groups with more
        terms are used first; a group with fewer terms is only scored when
        the slots are not yet full. Groups of skip_size terms are the AND
        matches, which search already returned.
        """
        # doc_id -> (indexes of the terms it contains, their postings).
        # posting_lists are read in order, so both lists stay in term order.
        found: dict[int, tuple[list[int], list[Posting]]] = {}
        for term_index, postings in enumerate(posting_lists):
            for posting in postings:
                entry = found.get(posting.doc_id)
                if entry is None:
                    found[posting.doc_id] = ([term_index], [posting])
                else:
                    entry[0].append(term_index)
                    entry[1].append(posting)

        # (term indexes) -> {doc_id: postings}: one group per set of terms.
        groups: dict[tuple[int, ...], dict[int, list[Posting]]] = {}
        for doc_id, (term_indexes, postings) in found.items():
            if len(term_indexes) == skip_size:
                continue  # an AND match, already in the results
            groups.setdefault(tuple(term_indexes), {})[doc_id] = postings

        results: list[Result] = []
        for size in range(len(posting_lists), 0, -1):
            if len(results) == needed:
                break
            # Rank each group of this size, then merge them by score.
            level: list[tuple[int, float]] = []
            for term_indexes, matches in groups.items():
                if len(term_indexes) == size:
                    group_dfs = [dfs[i] for i in term_indexes]
                    level += self._rank(matches, group_dfs, needed - len(results))
            level.sort(key=lambda item: (-item[1], item[0]))
            for doc_id, score in level[: needed - len(results)]:
                results.append(Result(doc_id, self.urls[doc_id], score, size))
        return results

    def close(self) -> None:
        """Close the index file."""
        self.reader.close()

    def __enter__(self) -> "Searcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
