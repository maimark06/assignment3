"""Interactive console interface for the search component."""

from time import perf_counter


def run_search(index_dir: str, k: int = 5) -> None:
    """Open a Searcher once and display results until the user exits.

    Searcher owns retrieval and ranking. Only its search call is timed;
    opening the index, waiting for input, and printing are excluded.
    """
    if k < 1:
        raise ValueError("The number of results must be at least 1")

    # Person C supplies this module; keep build/stats usable before it exists.
    from .searcher import Searcher

    with Searcher(index_dir) as searcher:
        print("Search ready. Type a query, or quit/exit to leave.")
        while True:
            try:
                query = input("Search> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                return

            if query.lower() in {"quit", "exit"}:
                print("Goodbye.")
                return
            if not query:
                continue

            start = perf_counter()
            results, total_matches = searcher.search(query, k=k)
            elapsed_ms = (perf_counter() - start) * 1000

            print(f"Matches: {total_matches:,} | Search time: {elapsed_ms:.2f} ms")
            if not results:
                print("No results found.")
            for rank, result in enumerate(results, start=1):
                print(f"{rank}. {result.url} (score: {result.score:.6f})")
            print()
