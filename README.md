# notion-proxy-search-mcp

A read-only Notion proxy with local hybrid search, spoken over MCP on stdio.

It keeps a SQLite mirror of the Notion pages an integration can see, indexes
them for both keyword and vector retrieval, and serves the result through tools
that stand in for the hosted Notion MCP server's rate-limited ones. Writes are
**not** mirrored back: an agent still edits Notion through the official server,
and the proxy notices the change on its next poll.

Everything is pure Go with no cgo — the embedding model runs in-process on
quantized int8 weights, and SQLite comes from `modernc.org/sqlite` — so the
whole thing is one static binary that cross-compiles to `linux/arm64` for a
Raspberry Pi.

## Why

Three things the hosted server cannot do:

- **No quota on reads.** Search, fetch and SQL answer from local disk.
- **SQL across data sources.** Every mirrored data source is a SQLite table
  named `collection://<data source id>`, with the same column conventions the
  official `notion-query-data-sources` tool uses, so existing queries run
  unchanged — and several of them can be joined in one statement.
- **Retrieval that is not keyword matching.** Embeddings plus BM25, fused by
  reciprocal rank, with one hop of relation-graph expansion.

The design reasoning, the Raspberry Pi sizing and the quantization choices are
in [docs/DESIGN.ru.md](docs/DESIGN.ru.md).

## Tools

| Tool | What it does |
| --- | --- |
| `notion-search` | Hybrid (vector + BM25) search with filters and optional relation expansion |
| `notion-fetch` | One page as Markdown with its properties, by id or URL |
| `notion-query-data-sources` | Read-only SQL over the mirrored data sources |
| `notion-list-data-sources` | Table names, columns and row counts — the schema needed to write SQL |
| `notion-recall` | Best match for a name plus every page linked to it, grouped by database |
| `notion-mirror-status` | Counts, embedding backlog, sync watermarks, recent log |
| `notion-resync` | Pull one page, or run a pass, right now instead of at the next poll |

## Processes

    notion-proxy-search-mcp serve     # MCP on stdio, one per client session
    notion-proxy-search-mcp daemon    # the mirror: sync loops, model, unix socket

`serve` reads the mirror file directly and asks the daemon only for what needs
the embedding model or the Notion token. It starts the daemon itself on first
use, so an MCP client only ever has to run `serve`. If the daemon is missing,
search degrades to keyword-only and says so in the response rather than failing.

One daemon holds one copy of the weights; the stdio processes hold none. That
split is what keeps a board with several concurrent sessions inside its memory
budget.

## Build

    make            # vet, test, build
    make arm64      # static linux/arm64 binary
    make model      # download the checkpoint and quantize it to build/model.npse
    make bench      # kernel throughput, worth re-running on the target board
    make bench-model NPS_TEST_MODEL=build/model.npse   # real forward-pass latency

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `NOTION_TOKEN` | — | Internal integration secret, read access. Daemon only |
| `NPS_DATA_DIR` | `/var/lib/notion-proxy-search-mcp` | Mirror, socket and daemon log |
| `NPS_MODEL` | `<data dir>/model.npse` | Weight blob; absent means keyword-only |
| `NPS_POLL_INTERVAL` | `30s` | Incremental sync period |
| `NPS_RECONCILE_INTERVAL` | `6h` | Full pass; this is what notices deletions |
| `NPS_EMBED_BATCH` | `8` | Chunks embedded per indexing tick |
| `NPS_EMBED_PAUSE` | `3s` | Pause between indexing ticks |
| `NPS_EMBED_WORKERS` | all cores | Goroutines for the embedding kernels |
| `NPS_EMBED_SKIP_SOURCES` | — | Data source ids left out of the vector index, comma separated; they stay searchable by keyword and SQL |
| `NPS_RPS` | `2.5` | Notion requests per second |
| `NPS_NOTION_VERSION` | `2025-09-03` | `Notion-Version` header |
| `NPS_NO_DAEMON` | — | `1` stops `serve` from starting the daemon |

The integration needs to be shared with the pages it should mirror: Notion grants
API access per page, and anything not shared is invisible to the proxy.

## Model

The default weight blob is `intfloat/multilingual-e5-small` quantized to int8:
12 layers, 384 dimensions, 100+ languages including Russian and Ukrainian, 118.7
MiB on disk and mapped read-only so several processes share one copy.

Any BERT-architecture sentence encoder with a Unigram SentencePiece tokenizer
converts the same way:

    notion-proxy-search-mcp convert -model-dir <checkpoint> -out model.npse -name <id>

The converter checks its own output: it loads the blob back and verifies that it
produces a unit-length vector before reporting success.

### How the engine is held to the reference

Reimplementing a tokenizer and an encoder is only defensible if the result is
measured against the originals, so three checks are committed. All three need a
weight blob; they skip without one.

    make model                                    # build build/model.npse
    NPS_TEST_MODEL=build/model.npse go test ./...

- **Tokenization** — `internal/embed/testdata/tokenizer_golden.json` holds 255
  reference segmentations from the Hugging Face tokenizer, across Russian,
  Ukrainian, Polish, English, SQL fragments, emoji and whitespace edge cases.
  The Go tokenizer matches all of them. Two divergences are deliberate and
  unit-tested: a NUL byte and NEL are dropped here, where the reference emits
  `<unk>` or its own piece.
- **Numerics** — `internal/embed/testdata/embedding_golden.json` holds vectors
  from onnxruntime running the checkpoint's own float32 ONNX export. The int8
  engine's worst cosine against it is 0.998, and the top-ranked candidate for
  every query is unchanged.
- **Retrieval** — `TestRetrievalQualityOnCorpus` measures hit@1 and hit@3 for
  each mode on a corpus you supply, so a ranking change can be judged instead of
  guessed:

      NPS_TEST_MODEL=build/model.npse \
      NPS_EVAL_FACTS=facts.json NPS_EVAL_QUESTIONS=questions.json \
      go test ./internal/index/ -run RetrievalQuality -v

  Regenerate the first two files with `scripts/tokenizer_reference.py` and
  `scripts/embedding_reference.py` after changing a model.
