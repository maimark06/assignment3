# Assignment 3 — Progress and Design Notes

This document records Person A's corpus and HTML work, then Person B's index
construction. The tokenizer will come from a teammate's earlier assignment and
has not been added yet.

> **Update (November 6, 2026):** `src/tokenizer.py` has since been added by
> Person C, and the full build now runs end to end on a small test corpus. The
> Person A notes below were written before that. See
> [Person C — Progress and Design Notes](#person-c--progress-and-design-notes)
> at the end of this file.

## Current progress

| Task | Status | Location |
| --- | --- | --- |
| Define the document data structure | Done | `src/corpus.py` |
| Read JSON pages and assign document IDs | Done | `src/corpus.py` |
| Remove URL fragments and skip duplicate URLs | Done | `src/corpus.py` |
| Skip invalid pages and log errors | Done | `src/corpus.py` |
| Extract text and important text from HTML | Done | `src/parser.py` |
| Count total and important token occurrences | Implemented; real tokenizer integration pending | `src/parser.py` |
| Add tokenizer code | Left to teammate | Expected: `src/tokenizer.py` |
| Add automated tests | 12 tests passing | `tests/test_corpus.py`, `tests/test_parser.py`, `tests/test_pipeline.py` |
| Declare HTML parser dependency | Done | `requirements.txt` |
| Keep the local planning file out of Git | Ignore rule added; tracking removed | `.gitignore` |
| Ignore datasets, generated indexes, and Python caches | Done | `.gitignore` |

No full build on the actual dataset has been verified yet. Person B has
implemented partial index construction and the merge. The CLI and the PDF
report belong to Person C.

## Data structure: `Document`

Location: `src/corpus.py`, class `Document`.

This dataclass represents one page and has three fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `doc_id` | `int` | Sequential ID starting at 0 |
| `url` | `str` | Page URL with its fragment removed |
| `html` | `str` | Content string read from the JSON file |

## Function: `iter_documents(data_dir)`

Location: `src/corpus.py`.

Signature:

```python
def iter_documents(data_dir: str) -> Iterator[Document]:
```

Recursively finds `.json` files under the given directory and yields one valid
`Document` at a time. It reads each page's `url` and `content` fields, strips the
URL fragment, skips duplicate URLs, and assigns consecutive IDs to accepted pages.

Important behavior:

- Paths are sorted before reading. IDs are reproducible for the same files and
  contents, but can change when files are added or removed.
- Fragment variants such as `page#one` and `page#two` share one document. The
  first valid page in sorted file order is kept.
- Empty HTML strings are allowed. Missing content or non-string content is rejected.
- Missing, blank, non-string, and fragment-only URLs are rejected.
- Invalid JSON, decoding errors, and file-reading errors are logged and skipped.
  Later valid pages can still be read, and skipped pages do not create ID gaps.
- A missing directory or a path that is not a directory raises `NotADirectoryError`
  when iteration begins.
- Files are read as UTF-8 with an optional BOM (`utf-8-sig`). The JSON `content`
  field is already a Unicode string, so the page's `encoding` metadata is not
  used to decode it again. Files that cannot be decoded are skipped; there is no
  fallback decoder for non-UTF-8 JSON files.
- Warning logs identify invalid files. An INFO summary reports accepted pages,
  invalid files, and duplicate URLs after the iterator is fully consumed.
  Configure logging at INFO level to see that summary. Counts are logged rather
  than returned as a separate result.
- HTML contents are read one page at a time, but the sorted file paths and the
  set of seen URLs are kept in memory.

## Function: `extract_text(html)`

Location: `src/parser.py`.

Signature:

```python
def extract_text(html: str) -> tuple[str, str]:
```

Returns `(all_text, important_text)`. Uses BeautifulSoup with Python's built-in
`html.parser`, which handles the tested plain-text and unclosed-tag inputs.

Processing steps:

1. Remove `script` and `style` elements and their contents.
2. Skip comments, declarations, and blank text nodes.
3. Add each remaining text node to the full text.
4. Also add it to important text if any ancestor is `b`, `strong`, `h1`, `h2`,
   `h3`, or `title`.
5. Join text pieces with spaces to avoid merging neighboring words.

Each text node is visited once. A word inside `<h1><strong>...</strong></h1>`
therefore appears once in important text, even though it has two important ancestors.
Title text is included in both outputs. Important text is a subset of the full text.

Example:

```python
from src.parser import extract_text

extract_text("<p>Hello</p><strong>world</strong>")
# ("Hello world", "world")
```

Empty or whitespace-only content returns `("", "")`. This function extracts text;
it does not lowercase, stem, or count words. It does not evaluate CSS visibility,
so text hidden by CSS may still be included.

## Function: `extract_tokens(html)`

Location: `src/parser.py`.

Signature:

```python
def extract_tokens(html: str) -> dict[str, tuple[int, int]]:
```

Calls `extract_text`, tokenizes the full text and important text separately, and
uses `Counter` to return:

```python
{stemmed_token: (total_frequency, important_frequency)}
```

The total already includes important occurrences. Do not add important frequency
to total frequency when building postings. A token with no important occurrences
gets an important frequency of 0.

For example, with a tokenizer that normalizes `cat` and `dog` unchanged:

```python
# HTML: <p>cat</p><strong>cat dog</strong>
# Result:
{"cat": (2, 1), "dog": (1, 1)}
```

### Tokenizer interface to confirm with the teammate

The current code expects `src/tokenizer.py` to expose:

```python
def tokenize(text: str):
    # Return an iterable of normalized, stemmed token strings.
    ...
```

The tokenizer should use alphanumeric sequences, lowercase them, apply Porter
stemming, preserve duplicate occurrences, and keep stop words. It must handle an
empty string by returning no tokens. Both calls must use the same normalization.
If the teammate's code returns a set or a frequency dictionary, adapt the interface
first: this function expects one token per occurrence.

The import happens inside `extract_tokens`, so `extract_text` can be used before
the tokenizer is added. Calling `extract_tokens` without that module currently
raises `ModuleNotFoundError`. Real stemming and end-to-end integration have not
been verified. The existing count test supplies temporary tokenizer results;
passing that test does not prove the real tokenizer is correct.

## Test functions

Tests use Python's standard-library `unittest`. Corpus tests create temporary
files and directories that are automatically cleaned up.

### `tests/test_corpus.py` — `IterDocumentsTests`

| Function | What it checks |
| --- | --- |
| `test_fragment_duplicates_keep_first_page_and_consecutive_ids` | Reads files inside a domain folder, strips fragments, keeps the first duplicate, and assigns IDs 0 and 1 without a gap. |
| `test_invalid_json_is_logged_and_skipped_without_interrupting_iteration` | Skips a broken JSON file between valid pages, records its filename in a warning, and reports one invalid file in the summary. |
| `test_invalid_page_fields_are_skipped` | Checks eight invalid record shapes or fields, then confirms a valid page with empty content is still accepted with ID 0. |
| `test_unicode_survives_invalid_file_encoding` | Preserves Unicode in UTF-8 and UTF-8-with-BOM files, skips an undecodable file, logs it, and continues with consecutive IDs. |

### `tests/test_parser.py` — `ExtractTextTests`

| Function | What it checks |
| --- | --- |
| `test_extract_tokens_counts_total_and_important_occurrences` | Uses a temporary tokenizer module to check total and important counts, plus the text passed to the tokenizer. It does not test stemming. |
| `test_nested_important_tags_count_each_occurrence_once` | Nested important tags do not duplicate text. |
| `test_unclosed_tags_preserve_text_and_importance` | Unclosed `h2` and `strong` tags still produce the expected full and important text. |
| `test_plain_text_has_no_important_text` | Plain text is preserved and important text is empty. Includes `ab12` and `3rd`, but does not test their tokenization. |
| `test_scripts_styles_and_comments_are_excluded` | Scripts, styles, and comments are excluded both outside and inside an important tag. |
| `test_empty_content_returns_empty_text` | Empty strings, whitespace, empty paragraphs, and empty important tags return two empty strings. |
| `test_each_required_tag_marks_text_as_important` | Each of `b`, `strong`, `h1`, `h2`, `h3`, and `title` contributes to important text. |

### `tests/test_pipeline.py` — `PipelineTests`

| Function | What it checks |
| --- | --- |
| `test_json_page_becomes_document_and_token_counts` | Reads a temporary JSON page from a domain folder, removes its URL fragment, extracts title and body text, filters a script, and checks final counts with a temporary tokenizer. Real tokenization and stemming remain untested. |

Run from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -B -m unittest discover -s tests -v
```

Current result: 12 tests pass. The project uses Python 3.9+ type annotations.
BeautifulSoup is declared as `beautifulsoup4>=4.10,<5`. No external HTML parser
package is required because the code uses `html.parser`.

## Git and handoff notes

- `.gitignore` contains `/CLAUDE.md`, and that file was removed from
  Git tracking while keeping the local copy. If previously committed, old copies
  may still exist in Git history.
- `.gitignore` also excludes the root `data/` and `index/` directories,
  `__pycache__/` directories, and compiled Python files (`*.py[cod]`). Keep
  datasets and generated indexes in those root directories. Ignore rules do
  not remove files that Git already tracks.
- The indexer can consume `iter_documents`, then call `extract_tokens(doc.html)`
  after tokenizer integration. The returned tuple maps to posting fields
  `tf` (total frequency) and `imp` (important frequency).
- `doc_id` and `url` come from the `Document`. Writing `doc_ids.tsv` belongs to
  the indexer; Person A's code does not write it.

## Remaining work for Person A

1. Get the teammate's tokenizer and confirm or adapt its interface.
2. Run real tokenizer tests for stemming, repeated words, stop words, empty text,
   and alphanumeric tokens such as `ab12` and `3rd`.
3. Test the complete JSON → HTML extraction → token counts pipeline with the real
   tokenizer. The existing pipeline test uses a tokenizer substitute.
4. Try the actual dataset and check page counts, skipped-file logs, and sample
   output. Dataset extraction and a full corpus run have not been verified.
5. Hand off the working pipeline to B and C after tokenizer integration.

## Latest progress — November 6, 2026

The code and tests that can be completed independently of the teammate's
tokenizer are finished for now. There is no need to keep adding small tests
before checking the real tokenizer and dataset.

The last completed items were:

- Added ignore rules for the dataset, generated index, and Python caches, and
  checked that the rules work.
- Added `test_json_page_becomes_document_and_token_counts` in
  `tests/test_pipeline.py`. It connects JSON loading, URL cleanup, HTML text
  extraction, and word counting using a temporary tokenizer substitute.
- Ran all 12 tests successfully and checked for whitespace errors with
  `git diff --check`.
- Documented the expected tokenizer interface so the teammate can add their code.

At the latest repository check, `src/tokenizer.py` and the `data/` directory were
still missing. The last status check made no code changes. Passing the current
tests does not mean that real tokenization or a full dataset run has been verified.

The next step is to add the teammate's tokenizer as `src/tokenizer.py`, check its
`tokenize(text)` interface, and run integration tests with it. If the tokenizer
is not ready yet, the dataset can be extracted into `data/` first so corpus loading
and HTML extraction can be checked on real pages. The full Person A handoff remains
pending until these checks are complete.

## Person B — Index construction

Person B turns token counts into on-disk index files. `src/posting.py` defines
one document's counts and the text line used by every index file.
`src/indexer.py` collects those counts in memory and periodically writes a
partial file. `src/merge.py` combines the partial files into the final index.
Person C's `IndexWriter` records byte offsets while that final file is written.

The indexer does not call `extract_tokens`. The caller passes
`{term: (tf, imp)}`, so these tests do not need the missing tokenizer.

| Task | Status | Location |
| --- | --- | --- |
| Represent one document's term counts | Done | `src/posting.py` |
| Encode and decode one index line | Done | `src/posting.py` |
| Collect an inverted index in memory | Done | `src/indexer.py` |
| Flush sorted partial indexes at least three times on a large corpus | Done | `src/indexer.py` |
| Write `doc_ids.tsv` | Done | `src/indexer.py` |
| Merge partials into `index.txt` and `offsets.tsv` | Done | `src/merge.py` |
| Tests for the line format, flush, and merge | Done | `tests/test_indexer.py`, `tests/test_merge.py` |

### `src/posting.py`

`Posting` is one term's counts in one document.

| Field | Type | Meaning |
| --- | --- | --- |
| `doc_id` | `int` | Document that contained the term |
| `tf` | `int` | Total occurrences in that document |
| `imp` | `int` | Occurrences inside `b`, `strong`, `h1`, `h2`, `h3`, or `title` |

`tf` already includes the important occurrences. `imp` is stored beside it for
later ranking and is not added into `tf`.

```python
def format_line(term: str, postings: list[Posting]) -> str:
def parse_line(line: str) -> tuple[str, list[Posting]]:
```

`format_line("cat", [Posting(0, 2, 1), Posting(2, 3, 0)])` returns
`cat\t0:2:1,2:3:0\n`. `parse_line` reads that line back into the term and the
same postings. A missing term, a missing tab, or an empty posting list raises
`ValueError`.

Decision: `format_line` keeps the posting order it is given. The flush and the
merge sort by `doc_id` before calling it. The same line format is used for a
partial file and for the final `index.txt`.

### `src/indexer.py`

`Indexer(out_dir, flush_every=15000)` keeps two structures in memory: a term
map, `_index`, and the accepted pages, `_documents`.

```python
def add_document(self, doc: Document, tokens: dict[str, tuple[int, int]]) -> None:
def flush(self) -> None:
```

`add_document` appends one `Posting` per token and remembers `(doc_id, url)`.
It does not write index lines. A page with no tokens is still remembered,
because Milestone 2 needs its URL. Every accepted page counts toward the flush
threshold. When the count reaches `flush_every`, `add_document` calls `flush`.

`flush` writes the current term map to `out_dir/partials/partial-000.txt`, then
`partial-001.txt`, and so on. Terms are alphabetical. Postings on a line are
ordered by `doc_id`, so a batch that received documents out of order still
produces a sorted line. After the file write succeeds, the term map is cleared.
The URL list is kept. Each flush rewrites `doc_ids.tsv` from that full list, in
the order pages were added, including pages with no tokens. A row is
`doc_id<TAB>url`.

`postings(term)` returns the postings still in memory. `documents`,
`partial_paths`, and `flush_count` are properties, so they are read without
parentheses. `documents` and `partial_paths` return copies. There is no setter.

Decisions:

- The batch size counts documents rather than measuring process memory. The
  default of 15,000 documents produces four partials for about 56,000 pages,
  which satisfies the requirement of at least three dumps. The split protects
  memory; one large file would fit on disk.
- An empty term map does not create a partial file, so a blank batch is not
  counted as one of those dumps. `flush()` on a new indexer creates nothing.
- A short final batch stays in memory. The caller calls `flush()` once after
  the last document.
- `partial_paths` lists the files the merge should read. `flush_count` counts
  only non-empty partials.

### `src/merge.py`

```python
def merge_partials(partial_paths: list[Path], index_dir: PathLike) -> None:
```

This reads the partial files together with `heapq.merge`, joins every partial's
postings for the same term, sorts those postings by `doc_id`, and writes one
line per term through `IndexWriter`. The writer produces `index.txt` and
`offsets.tsv`. For example, `cat` in `partial-000.txt` as documents 0 and 1,
and in `partial-001.txt` as document 3, becomes one line:
`cat	0:2:1,1:1:0,3:3:1`.

`_iter_terms` parses one partial. `_write_grouped` holds only the current
term's postings, then `_emit` writes that term before the next term starts.
Partial files are closed after the merge and are not deleted.

Decisions:

- The merge key is the term alone. When the same term appears in several
  partials, the earlier partial is emitted first. Sorting again by `doc_id`
  keeps a hand-built partial valid too.
- Only one term is held in memory, so the full index is not assembled in RAM.
- `IndexWriter` owns `index.txt` and the byte-offset table. Person B does not
  write a second copy of the final index. During search, Person C loads
  `offsets.tsv` into memory and leaves `index.txt` on disk.
- A test indexes the same four pages twice: once with `flush_every=2` and once
  in a single batch. Both merges produce the same `index.txt`, and
  `IndexReader.lookup("cat")` returns that term's three postings.

### Person B tests

| Test | What it checks |
| --- | --- |
| `tests/test_indexer.py` `InMemoryIndexTests` | `tf` is stored unchanged, later documents append in order, and a page with no tokens keeps its id and URL. |
| `tests/test_indexer.py` `PostingLineTests` | A line round-trips through `parse_line`, caller order is preserved, and an empty posting list is rejected. |
| `tests/test_indexer.py` `PartialFlushTests` | Six documents with a batch size of two produce three sorted partials, a short tail is written by an explicit flush, and an empty map creates no partial file. |
| `tests/test_merge.py` `test_several_partials_match_one_single_pass_index` | Merging several partials matches one single-batch index, and the offset table can seek to `cat`. |

Run from the repository root with the command in the Person A section. Person B's
tests use the standard library and do not need BeautifulSoup. No full run on
`developer.zip` has been done. The CLI and the PDF report remain Person C's work.

---

# Person C — Progress and Design Notes

This section records the work completed for Person C: the on-disk index format,
term-offset lookup, report statistics, and the command-line entry point. It also
covers `src/tokenizer.py`, which Person C wrote so the pipeline could run end to
end.

## Current progress

| Task | Status | Location |
| --- | --- | --- |
| Write the final index and the term-offset table | Done | `src/index_io.py` |
| Look up one term's postings by seeking on disk | Done | `src/index_io.py` |
| Tests for writing and lookup | 9 tests passing | `tests/test_index_io.py` |
| Compute the report numbers | Done (no test file, by choice) | `src/stats.py` |
| CLI: `build` and `stats` commands | Done | `src/main.py` |
| Tokenizer with Porter stemming | Done | `src/tokenizer.py` |
| Declare the `nltk` dependency | Done | `requirements.txt` |
| End-to-end run on a small synthetic corpus | Passed | (scratch data, not committed) |
| Full run on `developer.zip` | Not started (dataset not unzipped yet) | — |
| `README.md` run instructions | Not started | `README.md` |
| Spot-check script | Not started | — |
| PDF report | Not started | — |

All 31 tests in the repository pass (team-wide, including Person B's indexer and
merge tests).

## On-disk index layout

Every index directory (for example `index/`) contains:

| File | Written by | Format |
| --- | --- | --- |
| `index.txt` | Merge (B) through `IndexWriter` (C) | One line per term, sorted: `term<TAB>doc_id:tf:imp,doc_id:tf:imp,...` |
| `offsets.tsv` | `IndexWriter` (C) | `term<TAB>byte offset of that term's line in index.txt` |
| `doc_ids.tsv` | Indexer (B) | `doc_id<TAB>url` |
| `partials/partial-NNN.txt` | Indexer (B) | Same line format as `index.txt`, one file per flush |

The line format itself is defined once, in `posting.format_line` and
`posting.parse_line` (Person B). `index_io.py` reuses those functions.

Postings are sorted by `doc_id`. `tf` already includes important occurrences,
so `imp` is never added to `tf`.

## Module: `src/index_io.py`

Constants: `INDEX_FILE = "index.txt"`, `OFFSETS_FILE = "offsets.tsv"`,
`DOC_IDS_FILE = "doc_ids.tsv"`. Other modules import these names instead of
typing the file names again.

### Class: `IndexWriter(index_dir)`

Streams the final index to disk and records where each term's line starts.

| Method | What it does |
| --- | --- |
| `write(term, postings)` | Formats one term with `format_line` and writes it. |
| `write_line(line)` | Writes an already-formatted line (`term\t...\n`). |
| `term_count` | Number of terms written so far. |
| `close()` | Closes `index.txt` and writes `offsets.tsv`. |
| `abort()` | Closes `index.txt` without writing `offsets.tsv`. |

Use it as a context manager. Person B's `merge_partials` already does:

```python
with IndexWriter(out_dir) as writer:
    for term, postings in merged_terms:  # in sorted term order
        writer.write(term, postings)
```

Important behavior:

- Offsets are **byte** positions (the file is written in binary mode), so
  `seek()` lands on the right line even for non-ASCII terms.
- Terms must be **strictly increasing** in Python string order. A term that is
  out of order or repeated raises `ValueError`, because it means the merge has
  a bug.
- Malformed lines (no tab, no postings, missing or extra newline) raise
  `ValueError`.
- If an exception happens inside the `with` block, `offsets.tsv` is **not**
  written, so a crashed build never looks complete.

### Function: `load_offsets(index_dir) -> dict[str, int]`

Reads `offsets.tsv` into a dictionary from term to byte offset. This table is
the only part of the index held in memory during search. The posting lists stay
on disk, as the developer option requires.

### Class: `IndexReader(index_dir)`

Loads the offset table once and keeps `index.txt` open for repeated queries
(M2/M3).

| Method | What it does |
| --- | --- |
| `lookup(term)` | Seeks to the term's line, parses only that line, returns `list[Posting]`. Returns `[]` for a missing term. |
| `document_frequency(term)` | Length of the posting list (for idf in M2/M3). |
| `term in reader` | Whether the term is in the index. |
| `len(reader)` | Number of unique terms. |
| `close()` | Closes the file. Also works as a context manager. |

`lookup` expects a term that is already stemmed, so queries must go through
`tokenize` first. It also checks that the line it reads belongs to the
requested term. If `offsets.tsv` is from a different build, it raises
`ValueError` instead of returning the wrong postings.

### Function: `lookup(term, index_dir)`

One-off helper for scripts and debugging. It reloads the offset table on every
call, so search code should keep an `IndexReader` open instead.

### Tests: `tests/test_index_io.py`

| Function | What it checks |
| --- | --- |
| `test_writes_sorted_lines_and_offset_table` | One line per term; every offset points at the start of its term's line. |
| `test_write_line_accepts_preformatted_lines` | `write_line` works and the result can be looked up. |
| `test_rejects_out_of_order_and_duplicate_terms` | Unsorted and repeated terms raise `ValueError`. |
| `test_rejects_malformed_lines` | Missing tab, missing newline, empty term, or two lines in one call raise `ValueError`. |
| `test_no_offset_table_when_build_fails` | An exception inside the `with` block leaves no `offsets.tsv`. |
| `test_lookup_first_middle_last_and_non_ascii_terms` | Every term (including `café`) returns its own postings, reading forwards and backwards. |
| `test_missing_term_returns_empty_list` | Missing terms return `[]`; `in` works. |
| `test_document_frequency_is_posting_list_length` | Document frequency equals posting-list length. |
| `test_stale_offset_table_is_detected` | An offset that points at the wrong line raises `ValueError`. |

## Module: `src/stats.py`

Computes the Milestone 1 report numbers from a finished index directory.

| Function | Returns |
| --- | --- |
| `count_lines(path)` | Number of non-empty lines, read one line at a time. |
| `file_size_kb(path)` | Size of one file in KB. |
| `count_documents(index_dir)` | Lines in `doc_ids.tsv`. |
| `count_unique_tokens(index_dir)` | Lines in `index.txt` (one per term). |
| `index_size_kb(index_dir)` | Size of `index.txt` + `offsets.tsv` in KB. |
| `compute_stats(index_dir)` | An `IndexStats` dataclass with all of the numbers. |
| `format_stats(stats)` | The report table as text (also valid Markdown). |

How the numbers are measured (state this in the report):

- **Index size** = `index.txt` + `offsets.tsv`. `doc_ids.tsv` is shown on its
  own line and **not** added to the index size. Partial files are never counted.
- **1 KB = 1024 bytes.**
- `compute_stats` raises `FileNotFoundError` if a file is missing, and
  `ValueError` if `offsets.tsv` and `index.txt` have a different number of terms
  (they would come from different builds).

`IndexStats` uses plain fields only (`documents`, `unique_tokens`, `index_kb`,
`offsets_kb`, `total_index_kb`, `doc_ids_kb`); no `@property`.

## Module: `src/main.py`

Run from the repository root, inside the virtual environment:

```bash
source .venv/bin/activate
python -m src.main build --data data/ --out index/ [--flush-every 15000]
python -m src.main stats --index index/
```

`build` runs the whole pipeline:

1. `iter_documents` reads each JSON page (A).
2. `extract_tokens` turns the page into `{term: (tf, imp)}` (A + tokenizer).
3. `Indexer.add_document` collects postings and flushes a sorted partial file
   every `--flush-every` documents (B).
4. `Indexer.flush()` saves the last batch and `doc_ids.tsv` (B).
5. `merge_partials` merges the partials into `index.txt` + `offsets.tsv` (B,
   through `IndexWriter`).
6. Prints a summary and the stats table.

The summary includes documents indexed, documents skipped, number of partial
files, merge time, total build time, and peak memory (from `resource.getrusage`;
converted from bytes on macOS and from KB on Linux). A page whose parsing raises
an exception is logged, counted as skipped, and does not stop the build. A
warning is printed if fewer than 3 partial files were written.

With the default `--flush-every 15000`, about 55,000 documents should give 4
partial files.

`stats` prints the report table for an index that was already built and exits
with a clear message if files are missing.

Known issue: `check_required_modules()` is meant to stop the build with a
friendly message if `src/tokenizer.py` or `src/merge.py` is missing. For the
tokenizer it catches `ModuleNotFoundError`, but Python raises `ImportError`
there, so a missing tokenizer produces a traceback instead. Both files exist
now, so this does not affect current runs.

## Module: `src/tokenizer.py`

Signature, matching what `parser.extract_tokens` already calls:

```python
def tokenize(text: str) -> list[str]:
```

Rules:

- Lowercase the text, then take every run of `[a-z0-9]+` as a token. All other
  characters separate tokens, including apostrophes, hyphens, and non-ASCII
  letters.
- Porter-stem each token with nltk `PorterStemmer`.
- Keep stop words and keep duplicates (one token per occurrence).
- Empty or `None` text returns `[]`.
- A dictionary caches each word's stem so it is computed only once.

Examples:

```python
tokenize("Running dogs ran 3rd ab12 the e-mail Café MACHINE learning don't")
# ['run', 'dog', 'ran', '3rd', 'ab12', 'the', 'e', 'mail', 'caf', 'machin', 'learn', 'don', 't']
```

Team decision needed before the full build: non-ASCII letters split words
(`café` becomes `caf`). This follows the spec's "alphanumeric" wording
literally. If the team wants to keep Unicode letters, change `TOKEN_RE` before
building, since queries must use the same rule.

The search component (M2/M3) must call this same `tokenize` on queries.

`requirements.txt` now also contains `nltk>=3.8,<4`. Every team member must run
`pip install -r requirements.txt` again.

## End-to-end check on a synthetic corpus

The real dataset is not unzipped yet, so the pipeline was checked on 11 made-up
JSON files in 2 domain folders, built with `--flush-every 3`. The corpus
contained broken HTML, a plain-text page, an empty page, a `<script>` block, a
broken JSON file, and two URLs that differ only by fragment.

Results:

- 9 documents indexed. The broken JSON file and the duplicate-fragment URL were
  skipped and logged by `iter_documents`.
- 3 partial files were written and merged into `index.txt` with `offsets.tsv`.
- Terms that appear in several partials were merged correctly (for example
  `learn` from documents 0 and 5, `machin` from documents 0 and 8).
- Postings were checked by hand against the pages, for example:
  - `learn`: document 0 has `tf=2, imp=1` (once in `<title>`); document 5 has
    `tf=3`, with the `<script>` text excluded.
  - `acm`: `imp=1` both inside an unclosed `<strong>` and inside `<h2>`.
  - Stop words `the`, `of`, `and` are indexed. `don't` becomes `don`, `t`.
  - The empty page is still listed in `doc_ids.tsv`.
- `python -m src.main stats` reported 9 documents and 27 unique tokens.

## Git and handoff notes

- `src/main.py`, `src/tokenizer.py`, and the `requirements.txt` change were not
  yet committed at the time of writing.
- `.venv/` is not shown by `git status`, so it is not at risk of being committed.
- Person A owns `tokenizer.py` in the original plan. Tell Person A it now exists
  so they do not write a second one, and so they can update their notes and add
  real tokenizer tests.
- Run `git pull --rebase` (or set `git config pull.rebase true`) to avoid the
  "divergent branches" error when a teammate pushes while you have local
  commits.

## Remaining work for Person C

1. Commit and push `main.py`, `tokenizer.py`, and `requirements.txt`.
2. Unzip `developer.zip` into `data/` and run the full build. Record build time,
   number of partial files, peak memory, and skipped documents.
3. Check that the numbers are plausible: about 55,000 documents and a few hundred
   thousand to a few million unique tokens.
4. Write a spot-check script: for 5–10 terms, recount occurrences independently
   and compare with `IndexReader.lookup`.
5. Replace `README.md` with setup, build, stats, and test instructions.
6. Write the one-page PDF report: team names, the three required numbers, and
   the extras (partial files, build time, how size was measured, skipped
   documents).
