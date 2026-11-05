"""Extract page text and important text before tokenization."""

from collections import Counter

from bs4 import BeautifulSoup, CData, NavigableString


IMPORTANT_TAGS = frozenset({"b", "strong", "h1", "h2", "h3", "title"})


def extract_text(html: str) -> tuple[str, str]:
    """Return (all_text, important_text) without counting nested tags twice.

    Include title text in both outputs so it can contribute to indexing and
    ranking. The built-in HTML parser accepts plain text and broken markup.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Script and stylesheet contents are code, not searchable page text.
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    text_parts: list[str] = []
    important_parts: list[str] = []
    # Visit each text node once, even with several important ancestors.
    for node in soup.find_all(string=True):
        # Comments and declarations subclass NavigableString, so check the
        # exact type to keep them out of the searchable text.
        if type(node) not in (NavigableString, CData):
            continue
        text = node.strip()
        if not text:
            continue
        text_parts.append(text)
        if any(parent.name in IMPORTANT_TAGS for parent in node.parents):
            important_parts.append(text)

    return " ".join(text_parts), " ".join(important_parts)


def extract_tokens(html: str) -> dict[str, tuple[int, int]]:
    """Return each stemmed token's (total frequency, important frequency).

    The teammate-owned tokenizer must expose tokenize(text), yielding one
    normalized, stemmed token per occurrence, including duplicates and stop
    words. Important occurrences are included in the total frequency.
    """
    # Import here so extract_text remains usable before tokenizer integration.
    from .tokenizer import tokenize

    text, important_text = extract_text(html)
    total_counts = Counter(tokenize(text))
    important_counts = Counter(tokenize(important_text))

    return {
        token: (frequency, important_counts[token])
        for token, frequency in total_counts.items()
    }
