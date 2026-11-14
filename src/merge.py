"""Merge sorted partial indexes into one final index."""

import heapq
from pathlib import Path
from typing import Iterator, Optional, TextIO, Union

from .index_io import IndexWriter
from .posting import Posting, parse_line


PathLike = Union[str, Path]


def merge_partials(partial_paths: list[Path], index_dir: PathLike) -> None:
    """Stream partial indexes into index.txt and offsets.tsv.

    Each partial is already sorted by term. Postings for a term that was
    split across batches are combined and ordered by doc_id. Only the
    postings for the current term are held in memory.
    """
    files: list[TextIO] = []
    try:
        streams = []
        for path in partial_paths:
            source = Path(path).open(encoding="utf-8")
            files.append(source)
            streams.append(_iter_terms(source))
        with IndexWriter(index_dir) as writer:
            grouped = heapq.merge(*streams, key=lambda item: item[0])
            _write_grouped(writer, grouped)
    finally:
        for source in files:
            source.close()


def _iter_terms(source: TextIO) -> Iterator[tuple[str, list[Posting]]]:
    """Yield (term, postings) for every non-blank line in one partial."""
    for line in source:
        if line.strip():
            yield parse_line(line)


def _write_grouped(
    writer: IndexWriter,
    terms: Iterator[tuple[str, list[Posting]]],
) -> None:
    """Write one final line per term, joining postings from every partial."""
    current_term: Optional[str] = None
    current: list[Posting] = []
    for term, postings in terms:
        if term != current_term:
            _emit(writer, current_term, current)
            current_term = term
            current = list(postings)
        else:
            current.extend(postings)
    _emit(writer, current_term, current)


def _emit(writer: IndexWriter, term: Optional[str], postings: list[Posting]) -> None:
    """Write term once its postings from every partial have been collected."""
    if term is None:
        return
    writer.write(term, sorted(postings, key=lambda posting: posting.doc_id))
