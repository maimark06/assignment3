"""Collect an inverted index in memory before any disk flush."""

from pathlib import Path

from .corpus import Document
from .posting import Posting


class Indexer:
    """Map each term to the documents that contain it.

    out_dir and flush_every are stored for the later partial-index flush.
    This step only records postings and document URLs in memory.
    """

    def __init__(self, out_dir: str, flush_every: int = 15000) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self.out_dir = Path(out_dir)
        self.flush_every = flush_every
        self._index: dict[str, list[Posting]] = {}
        self._documents: list[tuple[int, str]] = []

    def add_document(self, doc: Document, tokens: dict[str, tuple[int, int]]) -> None:
        """Append one posting per token and remember the document URL.

        tokens maps a stemmed term to (tf, imp). A document with no tokens
        is still recorded so its id and URL can be written later.
        """
        self._documents.append((doc.doc_id, doc.url))
        for term, (tf, imp) in tokens.items():
            self._index.setdefault(term, []).append(Posting(doc.doc_id, tf, imp))

    def postings(self, term: str) -> list[Posting]:
        """Return a copy of the postings stored for term, in insertion order."""
        return list(self._index.get(term, ()))

    @property
    def documents(self) -> list[tuple[int, str]]:
        """Return (doc_id, url) pairs in the order documents were added."""
        return list(self._documents)
