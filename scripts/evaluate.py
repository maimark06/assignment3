"""Run a list of queries and record the top results for before/after comparison.

Run from the repository root, after building the index:

    python -m scripts.evaluate --index index/ --label baseline --out eval/baseline.txt
    ... change ranking, rebuild if needed ...
    python -m scripts.evaluate --index index/ --label after --out eval/after.txt
    diff eval/baseline.txt eval/after.txt

Queries come from scripts/queries.txt by default (one per line; blank lines
and # comments are ignored). For each query the report shows the number of
AND matches, the search time (first run and median), and the top k URLs with
their scores. Fill-in results that do not contain every query term are
marked "partial m/n terms".

The header records the run label, date, git commit and index details, which
TEST.txt needs. An existing --out file is never overwritten, so a saved
baseline cannot be lost by accident.
"""

import argparse
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from src.index_io import INDEX_FILE, OFFSETS_FILE
from src.query import parse_query
from src.searcher import Searcher


DEFAULT_QUERIES = Path(__file__).with_name("queries.txt")

# Developer-option target for one query.
TARGET_MS = 300

BYTES_PER_MB = 1024 * 1024


def read_queries(path) -> list[str]:
    """Queries from a text file: one per line, skipping blanks and # comments."""
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    if not queries:
        raise ValueError(f"No queries in {path}")
    return queries


def git_commit() -> str:
    """Short commit hash, plus a note when tracked files have local changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"
    return f"{commit} (with uncommitted changes)" if changes else commit


def run_query(searcher: Searcher, query: str, k: int, repeat: int):
    """Run one query `repeat` times; return (results, matches, first ms, median ms).

    The first run may be slower because the operating system has not cached
    that part of index.txt yet. Results come from the first run; every run
    returns the same results.
    """
    times_ms = []
    results, matches = [], 0
    for run in range(repeat):
        start = time.perf_counter()
        found, total = searcher.search(query, k=k)
        times_ms.append((time.perf_counter() - start) * 1000)
        if run == 0:
            results, matches = found, total
    return results, matches, times_ms[0], statistics.median(times_ms)


def evaluate(index_dir, queries: list[str], k: int, repeat: int, label: str) -> str:
    """Run every query and return the full report as text."""
    index_dir = Path(index_dir)
    rows = []
    with Searcher(index_dir) as searcher:
        doc_count = searcher.doc_count
        term_count = len(searcher.reader)
        for query in queries:
            rows.append((query, *run_query(searcher, query, k, repeat)))

    index_mb = sum(
        (index_dir / name).stat().st_size for name in (INDEX_FILE, OFFSETS_FILE)
    ) / BYTES_PER_MB

    lines = [
        f"Evaluation run: {label}",
        f"Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"Engine commit: {git_commit()}",
        f"Index: {index_dir} ({doc_count:,} documents, {term_count:,} terms, "
        f"{INDEX_FILE} + {OFFSETS_FILE} = {index_mb:.1f} MB)",
        f"Top {k} results, {repeat} runs per query",
        "Matches = pages containing every query term (AND). Results after the",
        "AND matches are partial matches, marked with how many terms they contain.",
        "",
    ]

    # One block per query: matches, timing, ranked URLs.
    for number, (query, results, matches, first_ms, median_ms) in enumerate(rows, 1):
        term_count_in_query = len(parse_query(query))
        lines.append(f"== {number:02d}. {query}")
        lines.append(
            f"Matches: {matches:,} | first {first_ms:.1f} ms | median {median_ms:.1f} ms"
        )
        if not results:
            lines.append("  (no results)")
        for rank, result in enumerate(results, 1):
            note = ""
            if result.matched_terms < term_count_in_query:
                note = f", partial {result.matched_terms}/{term_count_in_query} terms"
            lines.append(f"  {rank}. {result.url} (score {result.score:.4f}{note})")
        lines.append("")

    # Summary table for TEST.txt and the timing check.
    width = max(len("Query"), max(len(query) for query in queries))
    lines.append("Summary")
    lines.append(
        f"  {'#':>2}  {'Query':<{width}}  {'Matches':>8}  {'First ms':>8}  {'Median ms':>9}"
    )
    for number, (query, _, matches, first_ms, median_ms) in enumerate(rows, 1):
        flag = "  OVER 300 ms" if median_ms > TARGET_MS else ""
        lines.append(
            f"  {number:>2}  {query:<{width}}  {matches:>8,}  {first_ms:>8.1f}  "
            f"{median_ms:>9.1f}{flag}"
        )
    medians = [row[4] for row in rows]
    slow = sum(1 for median in medians if median > TARGET_MS)
    lines.append("")
    lines.append(
        f"Slowest median: {max(medians):.1f} ms | mean median: "
        f"{statistics.mean(medians):.1f} ms | queries over {TARGET_MS} ms: {slow}"
    )
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run evaluation queries and save the results")
    parser.add_argument("--index", required=True, help="folder with a built index")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES),
                        help="query file, one query per line (default scripts/queries.txt)")
    parser.add_argument("--top", type=int, default=5, help="results per query (default 5)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="runs per query for timing (default 3)")
    parser.add_argument("--label", default="run",
                        help="name for this run, e.g. baseline or after-title-boost")
    parser.add_argument("--out", help="also save the report to this file (never overwritten)")
    args = parser.parse_args(argv)
    if args.top < 1 or args.repeat < 1:
        parser.error("--top and --repeat must be at least 1")

    # Check before searching, so a long run is not wasted on a name clash.
    out = Path(args.out) if args.out else None
    if out is not None and out.exists():
        sys.exit(f"Error: {out} already exists; choose a new --out name")

    report = evaluate(args.index, read_queries(args.queries), args.top, args.repeat, args.label)
    print(report, end="")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"Error: {error}")
