"""Turn text into stemmed tokens for indexing (and, in M2/M3, for queries).

Rules from the assignment spec:
    - Tokens are all alphanumeric sequences (runs of a-z and 0-9 after
      lowercasing). Everything else separates tokens, including
      apostrophes, hyphens and non-ASCII letters.
    - Each token is Porter-stemmed (nltk PorterStemmer).
    - Stop words are never removed: "the" and "of" are indexed too.
    - One token is returned per occurrence (duplicates kept), because
      parser.extract_tokens counts them to get term frequencies.
    - Empty text returns [].

Examples: "ab12" and "3rd" are single tokens; "e-mail" -> "e", "mail";
"don't" -> "don", "t"; "café" -> "caf".

The search component must call this same tokenize() on queries so query
terms match the indexed terms.
"""

import re

from nltk.stem import PorterStemmer


# Text is lowercased before matching, so only lowercase letters are needed.
TOKEN_RE = re.compile(r"[a-z0-9]+")

_stemmer = PorterStemmer()

# Stemming is the slow part, and the same words repeat across ~56k pages,
# so remember each word's stem the first time it is computed.
_stem_cache: dict[str, str] = {}


def stem(word: str) -> str:
    """Return the Porter stem of an already-lowercased word, using the cache."""
    stemmed = _stem_cache.get(word)
    if stemmed is None:
        stemmed = _stemmer.stem(word)
        _stem_cache[word] = stemmed
    return stemmed


def tokenize(text: str) -> list[str]:
    """Return one lowercased, stemmed token per alphanumeric run in text.

    Example: tokenize("Running dogs ran 3rd") -> ["run", "dog", "ran", "3rd"]
    """
    if not text:
        return []
    words = TOKEN_RE.findall(text.lower())
    return [stem(word) for word in words]
