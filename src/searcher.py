"""Answer one query: parse it, read posting lists from disk, AND them, rank.

Query flow (owner in brackets):

    query text
      -> parse_query        stemmed, deduplicated terms          [A, query.py]
      -> missing term?      AND cannot match -> no results       [C, here]
      -> posting_size       order terms shortest list first      [C, index_io]
      -> IndexReader.lookup seek() to each term's line on disk   [C, index_io]
      -> intersect          docs that contain every term         [B, retrieval.py]
      -> rank               tf-idf + important boost, top k      [B, ranking.py]
      -> doc_id -> URL      from doc_ids.tsv                     [C, here]

Memory: only the offset table and the doc_id -> URL list stay in RAM.
Posting lists are read from index.txt for each query and then discarded,
so the full index is never loaded (developer-option rule).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .index_io import IndexReader, load_doc_urls
from .query import parse_query
from .ranking import rank
from .retrieval import intersect

PathLike = Union[str, Path]


# --- Search -----------------------------------------------------------------

@dataclass
class Result:
    """One ranked search result."""

    doc_id: int
    url: str
    score: float


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
        except Exception:
            self.reader.close()  # do not leave index.txt open on failure
            raise
        # N for idf. Comes from the same file as the URLs, so they agree.
        self.doc_count = len(self.urls)
        if self.doc_count == 0:
            self.reader.close()
            raise ValueError(f"No documents in {index_dir}; run the build first")

    def search(self, query: str, k: int = 5) -> tuple[list[Result], int]:
        """Return (top k results best first, total number of AND matches).

        Returns ([], 0) for a query with no terms, or when any term is not in
        the index (AND needs every term, so nothing can match).
        """
        if k < 1:
            raise ValueError("k must be at least 1")

        terms = parse_query(query)
        if not terms:
            return [], 0
        for term in terms:
            if term not in self.reader:
                return [], 0

        # Shortest posting list first: intersecting from the rarest term keeps
        # the running result small. posting_size reads no postings.
        terms.sort(key=self.reader.posting_size)

        posting_lists = [self.reader.lookup(term) for term in terms]
        # df of each term = length of its posting list, in the same order.
        dfs = [len(postings) for postings in posting_lists]

        matches = intersect(posting_lists)
        if not matches:
            return [], 0

        # Length damping needs doc_lengths.tsv. Until that file is loaded,
        # every page has the same length, so the length norm is 1.
        ranked = rank(
            matches,
            dfs,
            self.doc_count,
            k,
            lengths=[1] * self.doc_count,
            avg_length=1.0,
        )
        results = [Result(doc_id, self.urls[doc_id], score) for doc_id, score in ranked]
        return results, len(matches)

    def close(self) -> None:
        """Close the index file."""
        self.reader.close()

    def __enter__(self) -> "Searcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
