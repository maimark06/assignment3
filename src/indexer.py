"""Build an inverted index in memory and flush it in sorted partial files."""

from pathlib import Path

from .corpus import Document
from .posting import Posting, format_line


class Indexer:
    """Map each term to the documents that contain it.

    A full batch of flush_every documents is written to
    out_dir/partials and removed from memory. Call flush after the last
    document so a short final batch is saved too.
    """

    def __init__(self, out_dir: str, flush_every: int = 15000) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self.out_dir = Path(out_dir)
        self.flush_every = flush_every
        self._index: dict[str, list[Posting]] = {}
        self._documents: list[tuple[int, str]] = []
        self._docs_since_flush = 0
        self._partial_paths: list[Path] = []
        self._flush_count = 0

    def add_document(self, doc: Document, tokens: dict[str, tuple[int, int]]) -> None:
        """Append one posting per token and remember the document URL.

        tokens maps a stemmed term to (tf, imp). A document with no tokens
        is still recorded so its id and URL can be written later. Every
        document counts toward the flush threshold.
        """
        self._documents.append((doc.doc_id, doc.url))
        for term, (tf, imp) in tokens.items():
            self._index.setdefault(term, []).append(Posting(doc.doc_id, tf, imp))
        self._docs_since_flush += 1
        if self._docs_since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write the current batch, then drop its postings from memory.

        Terms are alphabetical and postings are ordered by doc_id. An empty
        term map creates no partial file. doc_ids.tsv is rewritten from every
        document accepted so far, including pages that contained no tokens.
        """
        if not self._index and not self._documents:
            return

        wrote_partial = False
        if self._index:
            partials = self.out_dir / "partials"
            partials.mkdir(parents=True, exist_ok=True)
            path = partials / f"partial-{self._flush_count:03d}.txt"
            text = "".join(
                format_line(
                    term,
                    sorted(self._index[term], key=lambda posting: posting.doc_id),
                )
                for term in sorted(self._index)
            )
            path.write_text(text, encoding="utf-8")
            self._partial_paths.append(path)
            wrote_partial = True

        self.out_dir.mkdir(parents=True, exist_ok=True)
        url_lines = "".join(f"{doc_id}\t{url}\n" for doc_id, url in self._documents)
        (self.out_dir / "doc_ids.tsv").write_text(url_lines, encoding="utf-8")
        self._index.clear()
        self._docs_since_flush = 0
        if wrote_partial:
            self._flush_count += 1

    def postings(self, term: str) -> list[Posting]:
        """Return a copy of the postings still in memory for term."""
        return list(self._index.get(term, ()))

    @property
    def documents(self) -> list[tuple[int, str]]:
        """Return (doc_id, url) pairs in the order documents were added."""
        return list(self._documents)

    @property
    def partial_paths(self) -> list[Path]:
        """Return the partial index files written so far, in flush order."""
        return list(self._partial_paths)

    @property
    def flush_count(self) -> int:
        """Return how many non-empty partial index files have been written."""
        return self._flush_count
