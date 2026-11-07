# Assignment 3 — Person A Progress and Design Notes

This document records the work completed for Person A: reading corpus files,
extracting HTML text, and preparing word counts for the indexer. The tokenizer
will come from a teammate's earlier assignment and has not been added yet.

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

No full build on the actual dataset has been verified yet. Index construction,
disk merging, the CLI, and the report belong to Persons B and C.

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
