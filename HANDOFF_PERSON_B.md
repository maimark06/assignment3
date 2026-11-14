# Milestone 2 handoff: Person C → Person B

Hi! I've finished and pushed the code part of my Milestone 2 tasks. What's left
is mostly **running things on the real index and writing the report**, and that
depends on your `retrieval.py` and `ranking.py`. This file lists everything you
need, in order.

Pull `main` first. Everything below is already on GitHub.

## What is already done (Person C)

| Task | Status | Where |
| --- | --- | --- |
| `posting_size`, `doc_count`, `load_doc_urls` | Done, tested | `src/index_io.py`, `tests/test_index_io.py` |
| `Searcher` (query → disk lookups → AND → rank → URLs) | Done, tested | `src/searcher.py` |
| End-to-end search tests | Done (2 ranking tests skip until `ranking.py` exists) | `tests/test_searcher.py` |
| Memory and speed measurement script | Done, **not yet run on the real index** | `scripts/measure_search.py` |
| README with build/stats/search/test instructions | Done | `README.md` |

Person A's `query.py`, console (`search_cli.py`) and `search` command are also in.

## What is left

1. **Your code:** `src/retrieval.py` and `src/ranking.py` (Step 1).
2. **Build the real index** (Step 2).
3. **Task 4:** run the 4 report queries and record the top 5 URLs (Step 3).
4. **Task 5:** run the measurement script and record memory and speed (Step 4).
5. **Task 6:** write the PDF report (Step 5).
6. Small cleanups (Step 6).

---

## Step 1: `retrieval.py` and `ranking.py` (must match `Searcher`)

`src/searcher.py` already calls these. Please keep these exact names and
signatures:

```python
# src/retrieval.py
def intersect(posting_lists: list[list[Posting]]) -> dict[int, list[Posting]]:
    """AND: doc_id -> one Posting per input list (same order as posting_lists),
    only for docs present in every list. Each input list is sorted by doc_id."""

# src/ranking.py
IMP_WEIGHT = 1

def rank(matches: dict[int, list[Posting]], dfs: list[int],
         n_docs: int, k: int) -> list[tuple[int, float]]:
    """Top-k (doc_id, score), highest score first, ties broken by smaller doc_id.
    dfs[i] is the df of the term behind matches[doc_id][i]."""
```

How `Searcher.search` calls them, so you know what you receive:

- `posting_lists` is already ordered **shortest list first** (by `posting_size`).
  Each list is sorted by `doc_id`. An empty list never reaches you, because a
  missing term returns no results before `intersect` is called.
- `dfs[i] == len(posting_lists[i])`, in the same order.
- `n_docs` is the number of documents (lines in `doc_ids.tsv`, about 53,792).
- `k` is at least 1.
- `rank` must return `(doc_id, score)` pairs, best first. `Searcher` maps each
  `doc_id` to its URL.

Scoring formula from the plan:

```
eff_tf   = tf + IMP_WEIGHT * imp
w(t, d)  = (1 + log10(eff_tf)) * log10(N / df_t)
score(d) = sum of w(t, d) over the query terms
```

Note: a term that appears in every document has `log10(N / df) = 0`. That's
expected, not a bug.

**Temporary stand-ins:** until your files exist, `searcher.py` uses two small
stand-ins at the top of the file (`_standin_intersect` and `_standin_rank`).
They give every result a score of 0 and order results by doc_id. As soon as
`src/retrieval.py` and `src/ranking.py` exist, `Searcher` imports and uses
yours automatically, with no change needed in `searcher.py`.

**Check it works:**

```bash
python -m unittest discover -s tests -v
```

The two tests in `tests/test_searcher.py` that were skipped
(`test_scores_are_sorted_best_first` and
`test_important_tag_page_outranks_same_tf_plain_page`) will now run. Both must
pass.

---

## Step 2: build the real index

You need the unzipped `developer.zip` (a `DEV/` folder with 88 domain folders).

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main build --data path/to/DEV --out index/
```

- This takes a while (likely 15–45 minutes). Let it finish.
- `index/` is ignored by Git, so it won't be committed.
- Check the printed summary: about 53,792 documents, 4 partial files, about
  1,066,417 unique tokens. These should match the Milestone 1 numbers.

---

## Step 3: Task 4, the 4 report queries

No new code is needed. Use the console:

```bash
python -m src.main search --index index/ --top 5
```

Type each query, then `quit`:

1. `cristina lopes`
2. `machine learning`
3. `ACM`
4. `master of software engineering`

For each query, record:

- the top 5 URLs, in order
- the number of matches and the search time (ms) printed above the results

**Check the results make sense:**

- `cristina lopes` should return her ICS pages.
- `machine learning` should return ML-related course or research pages.
- `ACM` should return ACM chapter or ACM-related pages.
- `master of software engineering` should return MSWE program pages.

If they look wrong, the first thing to try is `IMP_WEIGHT` in `ranking.py`.

**Take a screenshot** of the terminal while one query is shown. The report
needs it.

---

## Step 4: Task 5, memory and speed

```bash
python -m scripts.measure_search --index index/ > measure.txt
```

It prints:

- peak memory after imports, after opening the index, and after running queries
- the size of `index.txt`
- the time per query
- a "Checks" section at the bottom

What to look at in "Checks":

- **Search memory vs `index.txt` size.** The rule is that search must not load
  the whole index. The offset table (about 1 million terms in a Python dict) may
  use 100–200 MB, which could be close to or above the size of `index.txt`. If
  so, **don't try to fix it for Milestone 2**. Write the number down honestly;
  it's a Milestone 3 job. A possible fix is keeping only every 100th term in
  memory.
- **Slowest query vs 300 ms.** `the` and `of the and` are the worst cases on
  purpose. The 300 ms target is for Milestone 3, so just record it.

Don't commit `measure.txt`. Copy its numbers into the report.

---

## Step 5: Task 6, the PDF report

Must have:

- [ ] All three team members' names
- [ ] For each of the 4 queries: a table with ranks 1–5 and their URLs
- [ ] A screenshot of the console search running one query

Nice to have (one or two pages total is fine):

- [ ] Matches and time (ms) per query
- [ ] One line on ranking: "AND matches ranked by tf-idf, where occurrences in
      title/h1–h3/b/strong count extra (IMP_WEIGHT = 1)"
- [ ] Peak search memory vs `index.txt` size, from Step 4

Table template for each query:

```
Query: cristina lopes   (N matches, X ms)
| Rank | URL |
|------|-----|
| 1    |     |
| 2    |     |
| 3    |     |
| 4    |     |
| 5    |     |
```

Submit on Canvas **before** the deadline. Please send the PDF to the group
first so we can all check it.

---

## Step 6: small cleanups (after Step 1 works)

- [ ] In `src/searcher.py`, delete the stand-in block (`_standin_intersect`,
      `_standin_rank`, and the two `try/except` imports) and replace it with
      plain imports:
      ```python
      from .ranking import rank
      from .retrieval import intersect
      ```
      Then run the tests again.
- [ ] In `README.md`, add your two files to the "Project layout" list:
      ```
      retrieval.py    AND intersection of doc_id-sorted posting lists
      ranking.py      tf-idf + important-tag boost, top k
      ```
- [ ] Update your section in `DOCUMENT.md` if you're keeping notes there.

---

## Quick checklist

- [ ] `retrieval.py` + `ranking.py` pushed, all tests pass (no skips in `test_searcher.py`)
- [ ] Real index built
- [ ] 4 queries run, top 5 URLs recorded, screenshot taken
- [ ] `measure_search` run, numbers recorded
- [ ] Stand-ins removed from `searcher.py`, README layout updated
- [ ] PDF report shared with the group, then submitted on Canvas

Message me if anything in `searcher.py`, `index_io.py` or the measurement
script doesn't behave as described here.
