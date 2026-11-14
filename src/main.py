"""Command-line entry point for building the index and printing its stats.

Run from the repository root:

    python3 -m src.main build --data data/ --out index/
    python3 -m src.main stats --index index/

build pipeline:
    corpus.iter_documents  -> one Document per JSON page        (Person A)
    parser.extract_tokens  -> {term: (tf, imp)} for that page    (Person A)
    indexer.Indexer        -> postings in memory, flushed to
                              sorted partial files on disk       (Person B)
    merge.merge_partials   -> one final index.txt + offsets.tsv  (Person B,
                              writing through index_io.IndexWriter)
    stats.compute_stats    -> numbers for the report             (Person C)
"""

import argparse
import logging
import resource
import sys
import time

from .corpus import iter_documents
from .indexer import Indexer
from .parser import extract_tokens
from .stats import compute_stats, format_stats


# The developer option requires at least this many partial-index flushes.
MIN_PARTIAL_FLUSHES = 3

# Print a progress line after this many documents.
PROGRESS_EVERY = 5000

log = logging.getLogger("main")


def check_required_modules():
    """Stop early if a teammate's module is not in the repo yet.

    Without this check, a missing tokenizer would make every document fail
    one by one, and the build would "succeed" with an empty index.
    """
    missing = []
    try:
        from . import tokenizer  # noqa: F401  (Person A)
    except ModuleNotFoundError:
        missing.append("src/tokenizer.py (tokenize function)")
    try:
        from .merge import merge_partials  # noqa: F401  (Person B)
    except (ModuleNotFoundError, ImportError):
        missing.append("src/merge.py (merge_partials function)")
    if missing:
        sys.exit("Cannot build yet. Missing: " + "; ".join(missing))


def peak_memory_mb():
    """Peak memory (resident set size) used by this process, in MB.

    ru_maxrss is reported in bytes on macOS but in kilobytes on Linux.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def run_build(data_dir, out_dir, flush_every):
    """Build the full index from data_dir into out_dir and print a summary."""
    check_required_modules()
    from .merge import merge_partials

    start = time.perf_counter()
    indexer = Indexer(out_dir, flush_every=flush_every)
    indexed = 0
    skipped = 0

    # Pass 1: tokenize every page; the indexer flushes partial files on its own
    # every flush_every documents.
    for doc in iter_documents(data_dir):
        try:
            tokens = extract_tokens(doc.html)
        except Exception as error:
            # One bad page must not stop a 56k-page build. Log it and move on.
            skipped += 1
            log.warning("Skipped doc %d (%s): %s", doc.doc_id, doc.url, error)
            continue
        indexer.add_document(doc, tokens)
        indexed += 1
        if indexed % PROGRESS_EVERY == 0:
            log.info("Indexed %d documents, %d partial files so far",
                     indexed, indexer.flush_count)

    # Save the last, smaller batch and the final doc_ids.tsv.
    indexer.flush()
    partials = indexer.partial_paths
    log.info("Tokenizing done: %d documents, %d partial files",
             indexed, len(partials))
    if len(partials) < MIN_PARTIAL_FLUSHES:
        log.warning(
            "Only %d partial index files were written; the spec requires at "
            "least %d. Use a smaller --flush-every.",
            len(partials), MIN_PARTIAL_FLUSHES,
        )

    # Pass 2: merge all partial files into index.txt + offsets.tsv.
    merge_start = time.perf_counter()
    merge_partials(partials, out_dir)
    merge_seconds = time.perf_counter() - merge_start
    total_seconds = time.perf_counter() - start

    # Summary for the report.
    print()
    print(f"Documents indexed:      {indexed:,}")
    print(f"Documents skipped:      {skipped:,} (parse errors; see warnings above)")
    print(f"Partial index files:    {len(partials)} (flush every {flush_every:,} docs)")
    print(f"Merge time:             {merge_seconds:,.1f} s")
    print(f"Total build time:       {total_seconds:,.1f} s")
    print(f"Peak memory:            {peak_memory_mb():,.1f} MB")
    print()
    print(format_stats(compute_stats(out_dir)))


def run_stats(index_dir):
    """Print the report table for an index that was already built."""
    try:
        print(format_stats(compute_stats(index_dir)))
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"Error: {error}")


def main(argv=None):
    """Parse command-line arguments and run the chosen command."""
    parser = argparse.ArgumentParser(description="CS121 Assignment 3 indexer")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build the index from the corpus")
    build.add_argument("--data", required=True,
                       help="folder with the unzipped JSON pages, e.g. data/")
    build.add_argument("--out", required=True,
                       help="folder to write the index into, e.g. index/")
    build.add_argument("--flush-every", type=int, default=15000,
                       help="documents per partial index file (default 15000, "
                            "about 4 partials for the developer corpus)")

    stats = commands.add_parser("stats", help="print the report numbers")
    stats.add_argument("--index", required=True,
                       help="folder containing a built index, e.g. index/")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "build":
        run_build(args.data, args.out, args.flush_every)
    else:
        run_stats(args.index)


if __name__ == "__main__":
    main()
