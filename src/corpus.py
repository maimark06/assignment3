"""Read crawled JSON pages one at a time for the indexer."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urldefrag


logger = logging.getLogger(__name__)


@dataclass
class Document:
    doc_id: int
    url: str
    html: str


def iter_documents(data_dir: str) -> Iterator[Document]:
    """Yield valid pages with consecutive IDs; log and count invalid files.

    JSON files are UTF-8 (an optional BOM is accepted). Their content field
    is already Unicode after JSON decoding: the page's encoding metadata
    must not be used to decode that string a second time. Fragment variants
    of a URL are treated as one document, keeping the first valid page.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus directory does not exist: {root}")

    seen_urls: set[str] = set()
    doc_id = 0
    skipped = 0
    duplicates = 0

    # Sort paths so IDs are reproducible across runs with the same corpus.
    for path in sorted(root.rglob("*.json")):
        try:
            with path.open(encoding="utf-8-sig") as source:
                page = json.load(source)
            if not isinstance(page, dict):
                raise ValueError("Page must be a JSON object")
            url = page.get("url")
            html = page.get("content")
            if not isinstance(url, str) or not url.strip():
                raise ValueError("Page must contain a nonempty string URL")
            if not isinstance(html, str):
                raise ValueError("Page content must be a string")
            url, _ = urldefrag(url)
            if not url.strip():
                raise ValueError("URL is empty after removing its fragment")
        except (OSError, UnicodeError, ValueError) as error:
            skipped += 1
            logger.warning("Skipping %s: %s", path, error)
            continue

        if url in seen_urls:
            duplicates += 1
            continue
        seen_urls.add(url)
        yield Document(doc_id=doc_id, url=url, html=html)
        doc_id += 1

    logger.info(
        "Corpus complete: %d documents, %d invalid files skipped, "
        "%d duplicate URLs skipped",
        doc_id, skipped, duplicates,
    )
