"""Prepare search queries using the same tokenizer as the indexer."""

from .tokenizer import tokenize


def parse_query(text: str) -> list[str]:
    """Return unique stemmed terms in their first-occurrence order.

    AND retrieval only needs each normalized term once. Keep stop words and
    use the indexing tokenizer so query terms match terms stored on disk.
    """
    return list(dict.fromkeys(tokenize(text)))
