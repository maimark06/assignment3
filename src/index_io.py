"""Write the final index with a term-offset table, and read postings by seeking.

On-disk layout inside an index directory:

    index.txt    one line per term, sorted by term:
                 term<TAB>doc_id:tf:imp,doc_id:tf:imp,...
                 (line encoding is defined by posting.format_line/parse_line)
    offsets.tsv  term<TAB>byte offset of that term's line in index.txt
    doc_ids.tsv  doc_id<TAB>url (written by the indexer, not by this module)
    doc_lengths.tsv
                 doc_id<TAB>token count of that page (written by the indexer;
                 used for length damping in ranking)

Search must not load the whole index into memory (developer-option rule),
and since M3 its memory must stay below the index size. The full offset
table (about a million terms) is itself larger than the index once loaded
into a Python dict, so IndexReader keeps only a sparse table: every
SPARSE_EVERY-th term of offsets.tsv with that row's byte position in
offsets.tsv. A lookup then

    1. binary-searches the sparse table for the block the term falls in,
    2. seeks into offsets.tsv and reads at most SPARSE_EVERY short rows
       to find the term's byte offset in index.txt,
    3. seeks into index.txt and parses just that one posting list.
"""

from bisect import bisect_right
from pathlib import Path
from typing import Optional, Union

from .posting import Posting, format_line, parse_line


# File names shared by the indexer, merge, stats and search code.
INDEX_FILE = "index.txt"
OFFSETS_FILE = "offsets.tsv"
DOC_IDS_FILE = "doc_ids.tsv"
DOC_LENGTHS_FILE = "doc_lengths.tsv"

# IndexReader keeps one of every SPARSE_EVERY terms in memory. 100 keeps
# about 10.7k entries for 1.07M terms, and a lookup scans at most 100 rows
# of offsets.tsv (roughly 1.5 KB), which takes well under a millisecond.
SPARSE_EVERY = 100

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
    """Load the whole term-offset table into a dict (term -> byte offset).

    For tests and debugging only. On the full corpus this dict is larger
    than the index itself, so search uses IndexReader's sparse table.
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


def count_documents(index_dir: PathLike) -> int:
    """Number of documents (non-empty lines in doc_ids.tsv); 0 if it is missing.

    A missing file gives 0 instead of an error so an IndexReader can still be
    opened on an index written without doc_ids.tsv (as some tests do).
    Search code must treat 0 documents as "no index built".
    """
    path = Path(index_dir) / DOC_IDS_FILE
    if not path.is_file():
        return 0
    count = 0
    with open(path, "rb") as f:
        for line in f:  # one line at a time, never the whole file
            if line.strip():
                count += 1
    return count


def load_doc_urls(index_dir: PathLike) -> list[str]:
    """Load doc_ids.tsv as a list where position i holds the URL of doc_id i.

    Allowed in memory during search: about 54k short strings. Raises
    ValueError if the ids are not consecutive from 0, because then a list
    position would no longer equal the doc_id.
    """
    urls: list[str] = []
    path = Path(index_dir) / DOC_IDS_FILE
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            doc_id, separator, url = line.partition("\t")
            if not separator or not url:
                raise ValueError(f"Bad line {line_number} in {path}: {line!r}")
            if int(doc_id) != len(urls):
                raise ValueError(
                    f"{path} line {line_number}: expected doc_id {len(urls)}, "
                    f"found {doc_id}"
                )
            urls.append(url)
    return urls


def load_doc_lengths(index_dir: PathLike) -> list[int]:
    """Load doc_lengths.tsv as a list where position i holds doc_id i's length.

    The length is the page's raw token count (stop words included). Like the
    URL list this is one small entry per page (about 54k ints), so it is
    allowed in memory during search. Raises ValueError if the ids are not
    consecutive from 0 or a length is not a non-negative whole number.
    """
    lengths: list[int] = []
    path = Path(index_dir) / DOC_LENGTHS_FILE
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            doc_id, separator, length = line.partition("\t")
            if not separator or not doc_id.isdigit() or not length.isdigit():
                raise ValueError(f"Bad line {line_number} in {path}: {line!r}")
            if int(doc_id) != len(lengths):
                raise ValueError(
                    f"{path} line {line_number}: expected doc_id {len(lengths)}, "
                    f"found {doc_id}"
                )
            lengths.append(int(length))
    return lengths


class IndexReader:
    """Look up posting lists by seeking into offsets.tsv and index.txt.

    Keeps a sparse offset table in memory and both files open, so repeated
    queries do not reopen files or reload the table:

        with IndexReader("index/") as reader:
            postings = reader.lookup("comput")

    Attributes:
        doc_count  number of documents (N for idf), from doc_ids.tsv
    """

    def __init__(self, index_dir: PathLike, every: int = SPARSE_EVERY) -> None:
        if every < 1:
            raise ValueError("every must be at least 1")
        self.index_dir = Path(index_dir)
        self.doc_count = count_documents(self.index_dir)
        self._every = every
        self._index_size = (self.index_dir / INDEX_FILE).stat().st_size

        # Sparse table: _block_terms[i] is the first term of block i (rows
        # i*every .. i*every+every-1 of offsets.tsv) and _block_positions[i]
        # is the byte position of that row in offsets.tsv.
        self._block_terms: list[str] = []
        self._block_positions: list[int] = []
        self._term_count = self._load_sparse_table()

        self._offsets_file = open(self.index_dir / OFFSETS_FILE, "rb")
        self._file = open(self.index_dir / INDEX_FILE, "rb")

    def _load_sparse_table(self) -> int:
        """Read offsets.tsv once, keep every Nth term; return the term count.

        Streams the file line by line, so the full table is never in memory.
        Also checks that terms are strictly increasing, because the binary
        search over blocks depends on that order.
        """
        path = self.index_dir / OFFSETS_FILE
        position = 0  # byte position of the current row in offsets.tsv
        previous: Optional[str] = None
        count = 0
        with open(path, "rb") as f:
            for raw in f:
                term, offset = _parse_offset_row(raw, path, count + 1)
                if previous is not None and term <= previous:
                    raise ValueError(
                        f"{path} row {count + 1}: terms must be strictly "
                        f"increasing ({term!r} after {previous!r})"
                    )
                if count % self._every == 0:
                    self._block_terms.append(term)
                    self._block_positions.append(position)
                previous = term
                position += len(raw)
                count += 1
        return count

    def _locate(self, term: str) -> Optional[tuple[int, int]]:
        """Return (start, end) byte range of term's line in index.txt, or None.

        Finds the block with a binary search in memory, then reads at most
        `every` rows of offsets.tsv, plus one more row for the end offset.
        """
        block = bisect_right(self._block_terms, term) - 1
        if block < 0:
            return None  # sorts before the first term in the index
        f = self._offsets_file
        f.seek(self._block_positions[block])
        row = (block * self._every) + 1  # row number, for error messages
        for _ in range(self._every):
            raw = f.readline()
            if not raw:
                return None  # end of file
            found, start = _parse_offset_row(raw, f.name, row)
            if found == term:
                # The line ends where the next term's line starts.
                next_raw = f.readline()
                if next_raw:
                    end = _parse_offset_row(next_raw, f.name, row + 1)[1]
                else:
                    end = self._index_size  # last line ends at end of file
                return start, end
            if found > term:
                return None  # passed where the term would be
            row += 1
        return None  # not in this block, and the next block starts after it

    def posting_size(self, term: str) -> int:
        """Byte length of term's line in index.txt, or 0 if term is absent.

        A longer line means a longer posting list, so this orders query terms
        shortest-first without reading or parsing any postings.
        """
        location = self._locate(term)
        if location is None:
            return 0
        start, end = location
        return end - start

    def lookup(self, term: str) -> list[Posting]:
        """Return the postings for an already-stemmed term, or [] if absent.

        The caller must normalize the term with the same tokenizer and
        stemmer used at index time.
        """
        location = self._locate(term)
        if location is None:
            return []
        offset = location[0]
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
        return self._locate(term) is not None

    def __len__(self) -> int:
        """Number of unique terms in the index."""
        return self._term_count

    def close(self) -> None:
        self._offsets_file.close()
        self._file.close()

    def __enter__(self) -> "IndexReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _parse_offset_row(raw: bytes, path, row: int) -> tuple[str, int]:
    """Split one offsets.tsv row (bytes) into (term, offset in index.txt)."""
    term, separator, offset = raw.decode("utf-8").rstrip("\n").partition("\t")
    if not separator or not term or not offset.isdigit():
        raise ValueError(f"Bad row {row} in {path}: {raw!r}")
    return term, int(offset)


def lookup(term: str, index_dir: PathLike) -> list[Posting]:
    """One-off lookup for scripts and debugging.

    Loads the offset table on every call, so search code should keep an
    IndexReader open instead.
    """
    with IndexReader(index_dir) as reader:
        return reader.lookup(term)
