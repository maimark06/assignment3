"""Compute the Milestone 1 report numbers from a finished index directory.

Report numbers and how each one is measured:

    documents      lines in doc_ids.tsv (one line per indexed document)
    unique tokens  lines in index.txt (one line per distinct stemmed term)
    index size     size of index.txt + offsets.tsv on disk, in KB

doc_ids.tsv is reported separately and is not part of the index size: it
maps ids to URLs for display and is not part of the inverted index itself.
Partial index files are temporary build output and are never counted.

KB means 1024 bytes throughout.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .index_io import DOC_IDS_FILE, INDEX_FILE, OFFSETS_FILE

PathLike = Union[str, Path]

BYTES_PER_KB = 1024


@dataclass
class IndexStats:
    """Numbers for the report table, all measured from files on disk."""

    documents: int        # lines in doc_ids.tsv
    unique_tokens: int    # lines in index.txt
    index_kb: float       # size of index.txt
    offsets_kb: float     # size of offsets.tsv
    total_index_kb: float  # index_kb + offsets_kb (the number in the report)
    doc_ids_kb: float     # size of doc_ids.tsv (reported separately)


def count_lines(path: PathLike) -> int:
    """Count non-empty lines in a file.

    Looping over a file object reads one line at a time, so even a large
    index file is never loaded into memory all at once. Blank lines are
    skipped so a stray empty line cannot inflate the numbers.
    """
    count = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def file_size_kb(path: PathLike) -> float:
    """Size of one file on disk, in KB."""
    return Path(path).stat().st_size / BYTES_PER_KB


def count_documents(index_dir: PathLike) -> int:
    """Number of indexed documents (lines in doc_ids.tsv)."""
    return count_lines(Path(index_dir) / DOC_IDS_FILE)


def count_unique_tokens(index_dir: PathLike) -> int:
    """Number of unique tokens (lines in index.txt, one per term)."""
    return count_lines(Path(index_dir) / INDEX_FILE)


def index_size_kb(index_dir: PathLike) -> float:
    """Index size on disk in KB: index.txt + offsets.tsv."""
    index_dir = Path(index_dir)
    return file_size_kb(index_dir / INDEX_FILE) + file_size_kb(index_dir / OFFSETS_FILE)


def compute_stats(index_dir: PathLike) -> IndexStats:
    """Measure every report number from the files in index_dir.

    Also checks that offsets.tsv has one entry per index line, since a
    mismatch means the two files come from different builds.
    """
    index_dir = Path(index_dir)
    for name in (INDEX_FILE, OFFSETS_FILE, DOC_IDS_FILE):
        if not (index_dir / name).is_file():
            raise FileNotFoundError(f"Missing {name} in {index_dir}; run the build first")

    unique_tokens = count_unique_tokens(index_dir)
    offset_entries = count_lines(index_dir / OFFSETS_FILE)
    if offset_entries != unique_tokens:
        raise ValueError(
            f"{OFFSETS_FILE} has {offset_entries} entries but {INDEX_FILE} has "
            f"{unique_tokens} terms; rebuild the index"
        )

    index_kb = file_size_kb(index_dir / INDEX_FILE)
    offsets_kb = file_size_kb(index_dir / OFFSETS_FILE)
    return IndexStats(
        documents=count_documents(index_dir),
        unique_tokens=unique_tokens,
        index_kb=index_kb,
        offsets_kb=offsets_kb,
        total_index_kb=index_kb + offsets_kb,
        doc_ids_kb=file_size_kb(index_dir / DOC_IDS_FILE),
    )


def format_stats(stats: IndexStats) -> str:
    """Render the report table as plain text (also valid Markdown)."""
    rows = [
        ("Number of indexed documents", f"{stats.documents:,}"),
        ("Number of unique tokens", f"{stats.unique_tokens:,}"),
        ("Total index size on disk (KB)", f"{stats.total_index_kb:,.2f}"),
        (f"  {INDEX_FILE} (KB)", f"{stats.index_kb:,.2f}"),
        (f"  {OFFSETS_FILE} (KB)", f"{stats.offsets_kb:,.2f}"),
        (f"{DOC_IDS_FILE}, not counted above (KB)", f"{stats.doc_ids_kb:,.2f}"),
    ]
    label_width = max(len(label) for label, _ in rows)
    value_width = max([len("Value")] + [len(value) for _, value in rows])
    lines = [
        f"| {'Metric'.ljust(label_width)} | {'Value'.rjust(value_width)} |",
        f"|{'-' * (label_width + 2)}|{'-' * (value_width + 1)}:|",
    ]
    for label, value in rows:
        lines.append(f"| {label.ljust(label_width)} | {value.rjust(value_width)} |")
    return "\n".join(lines)
