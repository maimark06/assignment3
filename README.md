# CS 121 Assignment 3 — Search Engine

A search engine for the UCI ICS web pages (developer corpus, about 55,000
pages), written from scratch in Python. No indexing libraries and no databases:
the index is a set of plain files on disk.

It has two programs:

- **Indexer** (`build`): reads the pages and writes an inverted index to disk.
- **Search** (`search`): a console prompt that answers queries by reading
  posting lists from disk.

## Requirements

- Python 3.9 or newer
- Packages in `requirements.txt` (BeautifulSoup for HTML, nltk for Porter stemming)

## Setup

Run every command from the repository root (the folder with this README).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Activate the virtual environment again in every new terminal before running
the commands below.

## Data

Unzip `developer.zip`. It contains a `DEV/` folder with one subfolder per
domain and one JSON file per page (`url`, `content`, `encoding`).

Either unzip it into `data/` inside the repository (ignored by Git):

```bash
unzip developer.zip -d data/
```

or leave it where it is and pass that path to `--data` below.

## 1. Build the index

```bash
python -m src.main build --data data/ --out index/
```

`--data` can be any folder containing the domain folders, for example
`~/Downloads/DEV`. JSON files are found recursively.

Optional: `--flush-every N` sets how many pages are indexed in memory before a
partial index is written to disk (default 15000, which gives about 4 partial
files for the developer corpus).

What happens:

1. Each page is read, its URL fragment (`#...`) is removed, and duplicate URLs
   are skipped. Broken JSON and broken HTML are logged and skipped, not fatal.
2. Text is extracted from the HTML (scripts and styles removed). Words inside
   `<title>`, `<h1>`–`<h3>`, `<b>` and `<strong>` are also counted as important.
3. Text is split into alphanumeric tokens, lowercased and Porter-stemmed.
   Stop words are kept.
4. Postings are collected in memory and written to a sorted partial index file
   every `--flush-every` pages.
5. The partial files are merged into the final index.

The build prints a summary: pages indexed and skipped, number of partial files,
build time, peak memory, and the report numbers. Expect it to take several
minutes or longer on the full corpus.

Output in `index/` (ignored by Git):

| File | Contents |
| --- | --- |
| `index.txt` | One line per term, sorted: `term<TAB>doc_id:tf:imp,doc_id:tf:imp,...` |
| `offsets.tsv` | `term<TAB>byte offset` of that term's line in `index.txt` |
| `doc_ids.tsv` | `doc_id<TAB>url` |
| `partials/partial-NNN.txt` | The partial indexes written during the build (same format as `index.txt`) |

In a posting, `tf` is how many times the term appears in the page, and `imp` is
how many of those occurrences are inside important tags (`imp` is already
included in `tf`).

## 2. Print the index statistics

```bash
python -m src.main stats --index index/
```

Prints the number of indexed documents, the number of unique tokens, and the
index size on disk in KB. The index size counts `index.txt` + `offsets.tsv`
(1 KB = 1024 bytes). `doc_ids.tsv` is listed separately and partial files are
not counted.

## 3. Search

```bash
python -m src.main search --index index/ --top 5
```

Type a query at the `Search>` prompt and press Enter. Type `quit` or `exit`
(or press Ctrl-D) to leave. `--top` sets how many results are shown (default 5).

Example session (the URLs depend on the index):

```
Search ready. Type a query, or quit/exit to leave.
Search> machine learning
Matches: 1,234 | Search time: 25.31 ms
1. https://... (score: 4.512345)
2. https://... (score: 4.398765)
...

Search> quit
Goodbye.
```

How a query is answered:

1. The query goes through the same tokenizer as the pages (lowercase, split on
   non-alphanumeric characters, Porter stem). `ACM` and `acm` give the same
   results. Repeated words count once.
2. **AND retrieval:** only pages that contain every query term are returned. If
   any term is not in the index, there are no results.
3. For each term, only its own line is read from `index.txt`, using its byte
   offset from `offsets.tsv`. The shortest posting list is read first.
4. Matching pages are ranked by tf-idf, with extra weight for occurrences in
   important tags, and the top results are printed with their scores.

Memory: search keeps only the term-offset table and the doc_id-to-URL list in
memory. Posting lists stay on disk and are read per query; the full index is
never loaded.

## 4. Measure search memory and speed

```bash
python -m scripts.measure_search --index index/
```

Runs the four report queries (`cristina lopes`, `machine learning`, `ACM`,
`master of software engineering`) plus worst-case common-word queries, then
prints:

- peak memory after imports, after opening the index, and after the queries
- the sizes of the index files
- matches and time (first run and median) per query
- whether search memory stays below the size of `index.txt`, and whether
  queries are within 300 ms

Options: `--repeat N` (runs per query, default 5), `--top K`,
`--no-worst-case` (only the four report queries).

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The tests build small indexes in temporary folders, so they do not need the
dataset or a built index.

## Project layout

```
src/
  corpus.py       read JSON pages, strip URL fragments, assign doc_ids
  parser.py       HTML -> all text + important text; per-page token counts
  tokenizer.py    alphanumeric tokens, lowercase, Porter stemming
  posting.py      Posting(doc_id, tf, imp) and the index line format
  indexer.py      in-memory index, flushed to sorted partial files
  merge.py        k-way merge of partial files into the final index
  index_io.py     write index.txt + offsets.tsv; seek-based lookup
  stats.py        report numbers (documents, unique tokens, size)
  query.py        query text -> stemmed, deduplicated terms
  searcher.py     query -> postings from disk -> AND -> rank -> URLs
  search_cli.py   console search prompt
  main.py         command line: build, stats, search
scripts/
  measure_search.py   memory and speed measurement
tests/            unit and end-to-end tests (unittest)
```
