"""Measure search memory and speed on a built index (Milestone 2, Task 5).

Run from the repository root, after building the index:

    python -m scripts.measure_search --index index/

What it shows:
    1. Peak memory (RSS) at three points: after imports, after opening the
       Searcher (offset table + URL list), and after running the queries
       (posting lists read from disk). Each step's increase shows what that
       step costs.
    2. Time and number of AND matches for each query.
    3. Peak memory compared with the size of index.txt. Search must not load
       the whole index, so peak memory should stay well below that size.

Peak RSS from the operating system only ever goes up, so the three readings
show how much each step added on top of the previous one.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

from src.index_io import DOC_IDS_FILE, INDEX_FILE, OFFSETS_FILE
from src.main import peak_memory_mb
from src.searcher import Searcher


# The four queries the Milestone 2 report needs.
REPORT_QUERIES = [
    "cristina lopes",
    "machine learning",
    "ACM",
    "master of software engineering",
]

# Very common words have the longest posting lists, so they are the
# worst case for both memory and speed.
WORST_CASE_QUERIES = [
    "the",
    "of the and",
]

BYTES_PER_MB = 1024 * 1024


def file_mb(path: Path) -> float:
    """Size of one file in MB, or 0 if it does not exist."""
    return path.stat().st_size / BYTES_PER_MB if path.is_file() else 0.0


def time_query(searcher: Searcher, query: str, k: int, repeat: int):
    """Run one query `repeat` times.

    Returns (first run ms, median ms, number of AND matches). The first run
    may be slower because the operating system has not cached that part of
    index.txt yet; later runs show the warm speed.
    """
    times_ms = []
    total_matches = 0
    for _ in range(repeat):
        start = time.perf_counter()
        _, total_matches = searcher.search(query, k=k)
        times_ms.append((time.perf_counter() - start) * 1000)
    return times_ms[0], statistics.median(times_ms), total_matches


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Measure search memory and speed")
    parser.add_argument("--index", required=True, help="folder with a built index")
    parser.add_argument("--top", type=int, default=5, help="results per query (default 5)")
    parser.add_argument("--repeat", type=int, default=5,
                        help="runs per query for timing (default 5)")
    parser.add_argument("--no-worst-case", action="store_true",
                        help="only run the 4 report queries")
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.top < 1:
        parser.error("--repeat and --top must be at least 1")

    index_dir = Path(args.index)
    queries = REPORT_QUERIES + ([] if args.no_worst_case else WORST_CASE_QUERIES)

    # Step 1: baseline after Python, nltk and BeautifulSoup are imported.
    baseline_mb = peak_memory_mb()

    # Step 2: open the index. Loads offsets.tsv and doc_ids.tsv only.
    start = time.perf_counter()
    searcher = Searcher(index_dir)
    open_seconds = time.perf_counter() - start
    after_open_mb = peak_memory_mb()

    # Step 3: run every query; posting lists are read from disk per query.
    rows = []
    try:
        for query in queries:
            first_ms, median_ms, matches = time_query(searcher, query, args.top, args.repeat)
            rows.append((query, matches, first_ms, median_ms))
        term_count = len(searcher.reader)
        doc_count = searcher.doc_count
    finally:
        searcher.close()
    after_queries_mb = peak_memory_mb()

    index_mb = file_mb(index_dir / INDEX_FILE)
    offsets_mb = file_mb(index_dir / OFFSETS_FILE)
    doc_ids_mb = file_mb(index_dir / DOC_IDS_FILE)

    # --- Report ----------------------------------------------------------
    print(f"Index: {index_dir}  ({doc_count:,} documents, {term_count:,} terms)")
    print(f"Opening the Searcher took {open_seconds:.2f} s")
    print()

    print("Memory (peak RSS, MB)")
    print(f"| {'Point':<34} | {'Peak':>8} | {'Added':>8} |")
    print(f"|{'-' * 36}|{'-' * 9}:|{'-' * 9}:|")
    print(f"| {'After imports (baseline)':<34} | {baseline_mb:8.1f} | {'':>8} |")
    print(f"| {'After opening Searcher':<34} | {after_open_mb:8.1f} | "
          f"{after_open_mb - baseline_mb:8.1f} |")
    print(f"| {'After running all queries':<34} | {after_queries_mb:8.1f} | "
          f"{after_queries_mb - after_open_mb:8.1f} |")
    print()

    print("Files on disk (MB)")
    print(f"  {INDEX_FILE:<12} {index_mb:8.1f}")
    print(f"  {OFFSETS_FILE:<12} {offsets_mb:8.1f}")
    print(f"  {DOC_IDS_FILE:<12} {doc_ids_mb:8.1f}")
    print()

    print(f"Queries (top {args.top}, {args.repeat} runs each)")
    width = max(len("Query"), max(len(query) for query in queries))
    print(f"| {'Query':<{width}} | {'Matches':>8} | {'First ms':>9} | {'Median ms':>9} |")
    print(f"|{'-' * (width + 2)}|{'-' * 9}:|{'-' * 10}:|{'-' * 10}:|")
    for query, matches, first_ms, median_ms in rows:
        print(f"| {query:<{width}} | {matches:>8,} | {first_ms:>9.1f} | {median_ms:>9.1f} |")
    print()

    # --- Checks ------------------------------------------------------------
    search_mb = after_queries_mb - baseline_mb  # memory the search itself added
    slowest_ms = max(median_ms for _, _, _, median_ms in rows)
    print("Checks")
    print(f"  Search added {search_mb:.1f} MB; {INDEX_FILE} is {index_mb:.1f} MB "
          f"-> {'OK, below' if search_mb < index_mb else 'NOT below'} the index size")
    print(f"  Total peak {after_queries_mb:.1f} MB (includes the {baseline_mb:.1f} MB "
          f"Python baseline) -> {'below' if after_queries_mb < index_mb else 'NOT below'} "
          f"the index size")
    print(f"  Slowest median query {slowest_ms:.1f} ms -> "
          f"{'within' if slowest_ms <= 300 else 'OVER'} the 300 ms MS3 target")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"Error: {error}")
