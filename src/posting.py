"""One document's counts for a single indexed term."""

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
