"""Write the final index with a term-offset table, and read postings by seeking.

On-disk layout inside an index directory:

    index.txt    one line per term, sorted by term:
                 term<TAB>doc_id:tf:imp,doc_id:tf:imp,...
                 (line encoding is defined by posting.format_line/parse_line)
    offsets.tsv  term<TAB>byte offset of that term's line in index.txt
    doc_ids.tsv  doc_id<TAB>url (written by the indexer, not by this module)

Search must not load the whole index into memory (developer-option rule).
Only the offset table is kept in memory; each lookup seeks to one line of
index.txt and parses just that posting list.
"""

from pathlib import Path
from typing import Optional, Union

from .posting import Posting, format_line, parse_line


# File names shared by the indexer, merge, stats and search code.
INDEX_FILE = "index.txt"
OFFSETS_FILE = "offsets.tsv"
DOC_IDS_FILE = "doc_ids.tsv"

PathLike = Union[str, Path]


class IndexWriter:
    """Stream the final merged index to disk and record each term's offset.

    Terms must arrive in strictly increasing order (Python string order, the
    same order the partial indexes are sorted in). This catches merge bugs
    such as a term written twice. Use it as a context manager:

        with IndexWriter("index/") as writer:
            writer.write("apple", [Posting(0, 2, 1)])
            writer.write_line("banana\\t3:1:0\\n")

    The offset table is written only when the block exits without an error,
    so a crashed build never leaves an offsets file that looks complete.
    """

    def __init__(self, index_dir: PathLike) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        # Binary mode so the offsets we record are exact byte positions,
        # which is what seek() needs (character counts differ for non-ASCII).
        self._file = open(self.index_dir / INDEX_FILE, "wb")
        self._offsets: list[tuple[str, int]] = []
        self._position = 0  # byte offset where the next line will start
        self._last_term: Optional[str] = None
        self._closed = False

    def write(self, term: str, postings: list[Posting]) -> None:
        """Write one term with its postings (already sorted by doc_id)."""
        self.write_line(format_line(term, postings))

    def write_line(self, line: str) -> None:
        """Write one already-formatted index line and remember its offset.

        Useful for the merge step, which may combine posting text without
        turning it back into Posting objects.
        """
        if self._closed:
            raise ValueError("IndexWriter is already closed")
        # A line must be exactly one record: "term<TAB>postings\n".
        if not line.endswith("\n") or "\n" in line[:-1]:
            raise ValueError(f"Index line must end with one newline: {line!r}")
        term, separator, body = line.partition("\t")
        if not separator or not term or not body.strip():
            raise ValueError(f"Index line needs a term and postings: {line!r}")
        if self._last_term is not None and term <= self._last_term:
            raise ValueError(
                f"Terms must be strictly increasing: {term!r} after {self._last_term!r}"
            )

        data = line.encode("utf-8")
        self._file.write(data)
        self._offsets.append((term, self._position))
        self._position += len(data)
        self._last_term = term

    @property
    def term_count(self) -> int:
        """Number of terms written so far (the unique-token count when done)."""
        return len(self._offsets)

    def close(self) -> None:
        """Finish index.txt and write offsets.tsv next to it."""
        if self._closed:
            return
        self._file.close()
        self._closed = True
        write_offsets(self.index_dir, self._offsets)

    def abort(self) -> None:
        """Close index.txt without writing an offset table (used on errors)."""
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> "IndexWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def write_offsets(index_dir: PathLike, offsets: list[tuple[str, int]]) -> None:
    """Write the term-offset table as term<TAB>offset lines, in term order."""
    path = Path(index_dir) / OFFSETS_FILE
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for term, offset in offsets:
            f.write(f"{term}\t{offset}\n")


def load_offsets(index_dir: PathLike) -> dict[str, int]:
    """Load the term-offset table into a dict (term -> byte offset).

    This is the only part of the index held in memory during search. It has
    one small entry per term, which is far smaller than the posting lists.
    """
    offsets: dict[str, int] = {}
    path = Path(index_dir) / OFFSETS_FILE
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            term, separator, offset = line.rstrip("\n").partition("\t")
            if not separator or not term:
                raise ValueError(f"Bad line {line_number} in {path}: {line!r}")
            offsets[term] = int(offset)
    return offsets


class IndexReader:
    """Look up posting lists by seeking into index.txt.

    Keeps the offset table in memory and one open file handle, so repeated
    queries (M2/M3) do not reopen the file or reload the table:

        with IndexReader("index/") as reader:
            postings = reader.lookup("comput")
    """

    def __init__(self, index_dir: PathLike) -> None:
        self.index_dir = Path(index_dir)
        self.offsets = load_offsets(self.index_dir)
        self._file = open(self.index_dir / INDEX_FILE, "rb")

    def lookup(self, term: str) -> list[Posting]:
        """Return the postings for an already-stemmed term, or [] if absent.

        The caller must normalize the term with the same tokenizer and
        stemmer used at index time.
        """
        offset = self.offsets.get(term)
        if offset is None:
            return []
        self._file.seek(offset)
        line = self._file.readline().decode("utf-8")
        found_term, postings = parse_line(line)
        # A mismatch means offsets.tsv does not belong to this index.txt
        # (for example, the index was rebuilt without rewriting offsets).
        if found_term != term:
            raise ValueError(
                f"Offset table is out of date: expected {term!r} at byte "
                f"{offset}, found {found_term!r}"
            )
        return postings

    def document_frequency(self, term: str) -> int:
        """Number of documents containing term (posting-list length), for idf."""
        return len(self.lookup(term))

    def __contains__(self, term: str) -> bool:
        return term in self.offsets

    def __len__(self) -> int:
        """Number of unique terms in the index."""
        return len(self.offsets)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "IndexReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def lookup(term: str, index_dir: PathLike) -> list[Posting]:
    """One-off lookup for scripts and debugging.

    Loads the offset table on every call, so search code should keep an
    IndexReader open instead.
    """
    with IndexReader(index_dir) as reader:
        return reader.lookup(term)
