"""Build an inverted index in memory and flush it in sorted partial files."""

from pathlib import Path

from .corpus import Document
from .posting import Posting, format_line

# A batch stops growing once it holds this many Posting objects, so one
# partial cannot keep millions of postings in RAM. Document flushes still
# apply through flush_every.
DEFAULT_FLUSH_POSTINGS = 500_000


class Indexer:
    """Map each term to the documents that contain it.

    A batch is written to out_dir/partials when it reaches flush_every
    documents or flush_postings postings, then those postings are dropped.
    Call flush after the last document so a short final batch is saved too.
    """

    def __init__(
        self,
        out_dir: str,
        flush_every: int = 15000,
        flush_postings: int = DEFAULT_FLUSH_POSTINGS,
    ) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        if flush_postings < 1:
            raise ValueError("flush_postings must be at least 1")
        self.out_dir = Path(out_dir)
        self.flush_every = flush_every
        self.flush_postings = flush_postings
        self._index: dict[str, list[Posting]] = {}
        self._documents: list[tuple[int, str]] = []
        self._token_counts: list[int] = []
        self._docs_since_flush = 0
        self._posting_count = 0
        self._partial_paths: list[Path] = []
        self._flush_count = 0

    def add_document(self, doc: Document, tokens: dict[str, tuple[int, int]]) -> None:
        """Append one posting per token and remember the document URL.

        tokens maps a stemmed term to (tf, imp). The stored length is the
        sum of tf values only: imp is a ranking boost, not extra tokens.
        A document with no tokens is still recorded so its id, URL, and a
        length of 0 can be written later. A full document batch or a full
        posting batch is flushed.
        """
        token_count = sum(tf for tf, _imp in tokens.values())
        self._documents.append((doc.doc_id, doc.url))
        self._token_counts.append(token_count)
        for term, (tf, imp) in tokens.items():
            self._index.setdefault(term, []).append(Posting(doc.doc_id, tf, imp))
        self._posting_count += len(tokens)
        self._docs_since_flush += 1
        if (
            self._docs_since_flush >= self.flush_every
            or self._posting_count >= self.flush_postings
        ):
            self.flush()

    def flush(self) -> None:
        """Write the current batch, then drop its postings from memory.

        Terms are alphabetical and postings are ordered by doc_id. An empty
        term map creates no partial file. doc_ids.tsv and doc_lengths.tsv are
        rewritten from every document accepted so far, including pages that
        contained no tokens.
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
        length_lines = "".join(
            f"{doc_id}\t{token_count}\n"
            for (doc_id, _url), token_count in zip(self._documents, self._token_counts)
        )
        (self.out_dir / "doc_lengths.tsv").write_text(length_lines, encoding="utf-8")
        self._index.clear()
        self._docs_since_flush = 0
        self._posting_count = 0
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
