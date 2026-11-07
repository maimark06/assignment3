"""One document's counts for a single indexed term, and its text line format."""

from dataclasses import dataclass


@dataclass
class Posting:
    """Term frequency and important-tag frequency for one document.

    tf is the total number of occurrences. It already includes occurrences
    inside important tags, so imp must not be added to tf again.
    """

    doc_id: int
    tf: int
    imp: int


def format_line(term: str, postings: list[Posting]) -> str:
    """Encode one term and its postings as a single index line.

    The caller supplies postings already ordered by doc_id. This function
    does not sort them. A term with no postings is not a valid line.
    """
    if not term or not postings:
        raise ValueError("A term line requires a term and at least one posting")
    body = ",".join(
        f"{posting.doc_id}:{posting.tf}:{posting.imp}" for posting in postings
    )
    return f"{term}\t{body}\n"


def parse_line(line: str) -> tuple[str, list[Posting]]:
    """Decode one index line into a term and its postings."""
    term, separator, body = line.rstrip("\r\n").partition("\t")
    if not separator or not term or not body:
        raise ValueError("A term line requires a term and at least one posting")
    postings: list[Posting] = []
    for part in body.split(","):
        doc_id, tf, imp = part.split(":")
        postings.append(Posting(int(doc_id), int(tf), int(imp)))
    return term, postings
