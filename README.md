# RAG Generator

Upload any set of documents at runtime and get a **question-answering app** over them: grounded answers with clickable citations, and an honest "I couldn't find that" when the documents don't say. Each document set becomes its own RAG app with its own chat page and API endpoint, and no code changes are needed.

<!-- Screenshot: save a chat screenshot as docs/screenshot.png and uncomment -->
<!-- ![Chat with citations](docs/screenshot.png) -->

| Brief requirement | How it's met |
|---|---|
| Accepts documents at runtime | Drag-and-drop in the web UI or `POST /api/collections/{id}/documents`. PDF, DOCX, Markdown, TXT, HTML |
| Creates a RAG application over those documents | Every document set becomes a **collection**: its own index, chat page (`/c/<id>`) and query endpoint |
| Grounded answers | Hybrid retrieval + reranking; the LLM answers only from numbered sources, cites every claim `[n]`, and refuses when the answer isn't there |
| Works with different document sets without code changes | Collections are data, not code; models and retrieval settings are config / per-collection |

**Evaluation on the bundled sample sets:** 21/21 answers correct and correctly cited, 6/6 out-of-scope questions refused, median latency 1.4 s ([details](#evaluation)).

---

## Quick start

You need **Python 3.11–3.13** and [uv](https://docs.astral.sh/uv/), or Docker.

```bash
git clone https://github.com/tejas-dharani/rag-generator.git && cd rag-generator
cp .env.example .env          # then put your OPENAI_API_KEY in .env
make install                  # uv sync
make run                      # http://localhost:8000
```

Or with Docker:

```bash
cp .env.example .env          # add your key
docker compose up --build     # http://localhost:8000
```

Then open http://localhost:8000, create an app, drop in some documents (or use the ones in [`sample_docs/`](sample_docs)) and ask questions.

The first local run downloads two small models (~200 MB total) into `data/models/`. The Docker image has them built in.

### No OpenAI key?

- **Run fully locally with [Ollama](https://ollama.com)** (free, no key): `ollama pull qwen2.5:7b`, then set `LLM_PROVIDER=ollama` and `LLM_MODEL=qwen2.5:7b` in `.env`. Embeddings and reranking are already local.
- **Use Claude:** `uv sync --extra anthropic`, then set `LLM_PROVIDER=anthropic`, `LLM_MODEL=<claude model>` and `ANTHROPIC_API_KEY`.
- **Tests need no key at all:** `make test`.

---

## How it works

```
 INGESTION (background task)                     QUERY
 ──────────────────────────                      ─────
 upload ─► parse ─► chunk ─► embed ─► index      question ─► (condense follow-up using chat history)
   │        │         │        │        │                         │
   │   PDF/DOCX/MD/   │   bge-small   Chroma (vectors)            ├─► vector search (Chroma) ─┐
   │   TXT/HTML with  │   (local)   + BM25 (keywords)             └─► BM25 keyword search ────┤
   │   page/heading   │             + SQLite (metadata)                                       ▼
   │   metadata       │                                          Reciprocal Rank Fusion → top 20
   │            ~400-token chunks,                                            ▼
   │            split on headings →                          cross-encoder rerank → top k (5)
   │            paragraphs → sentences                                        ▼
   ▼                                                     LLM with numbered <source> blocks
 dedup by SHA-256                                        → streamed answer with [n] citations
                                                           or "I couldn't find …"
```

### Design decisions

| Decision | Why |
|---|---|
| **Collections as the "generator" unit** | One index per document set keeps apps isolated. Asking the product manual about annual leave is refused, not answered from the HR handbook |
| **Local embeddings (`BAAI/bge-small-en-v1.5` via FastEmbed/ONNX)** | Free, offline, fast on CPU, no PyTorch. Reviewers need only one API key (or none, with Ollama) |
| **Hybrid search: vectors + BM25** | Small embedding models miss exact tokens such as error codes (`E7`), part numbers (`CF-200`) and figures; BM25 catches them. RRF merges the two rankings without comparing their incompatible scores |
| **Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)** | Scores each question/passage pair jointly, so the best chunks reach the LLM first |
| **The LLM, not a score threshold, decides "not found"** | The eval showed that a correct chunk scored 0.0007, while out-of-scope questions scored 0.00001–0.00006 and one scored 0.78. Reranker scores are too noisy to decide whether a question is answerable, so the top 3 chunks always reach the LLM and the prompt makes it refuse when they don't contain the answer. An empty collection still refuses without an LLM call |
| **~400-token, structure-aware chunks** | bge-small reads at most 512 tokens. Chunks never cross a page or heading, so every citation points to one file, page and section; the heading is prefixed to each chunk to help retrieval |
| **Sources wrapped in `<source>` tags and marked untrusted** | Basic prompt-injection defence: instructions inside documents are treated as data |
| **One OpenAI-compatible client for OpenAI and Ollama** | Switching provider is a config change; Anthropic is an optional adapter |
| **Follow-up condensing** | "And for day trips?" is rewritten into a standalone question before retrieval, so chat history works with search |
| **Plain HTML/JS UI served by FastAPI** | No build step, one process, one port |
| **Background ingestion with status** | Uploads return immediately; each document shows `processing → ready / failed` with the error (e.g. scanned PDFs with no text) |

---

## Using the API

Interactive docs: http://localhost:8000/docs

```bash
# 1. Create a collection (optionally override chunk_size, chunk_overlap, top_k)
curl -s -X POST localhost:8000/api/collections \
  -H 'Content-Type: application/json' -d '{"name": "Product Manual"}'
# -> {"id": "1e4f0070893d", ...}

# 2. Upload documents (indexed in the background)
curl -s -X POST localhost:8000/api/collections/1e4f0070893d/documents \
  -F files=@sample_docs/product_manual/warranty.txt \
  -F files=@sample_docs/product_manual/setup_guide.pdf

# 3. Check ingestion status
curl -s localhost:8000/api/collections/1e4f0070893d/documents

# 4. Ask a question (add "stream": true for Server-Sent Events)
curl -s -X POST localhost:8000/api/collections/1e4f0070893d/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How long is the warranty if I register the product?"}'
```

The query response has this shape:

```json
{
  "answer": "If you register the product within 30 days of purchase, the main unit warranty is extended to **5 years**. [1]",
  "cited": [1],
  "grounded": true,
  "sources": [{"number": 1, "filename": "warranty.txt", "page": null, "heading": null, "text": "…", "score": 0.97}],
  "search_query": "How long is the warranty if I register the product?"
}
```

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Status and configured models |
| `POST` / `GET` | `/api/collections` | Create / list collections |
| `GET` / `DELETE` | `/api/collections/{id}` | Collection details / delete |
| `POST` / `GET` | `/api/collections/{id}/documents` | Upload files / list with status |
| `DELETE` | `/api/collections/{id}/documents/{doc_id}` | Remove a document |
| `POST` | `/api/collections/{id}/query` | Ask (`question`, optional `history`, `stream`) |

---

## Evaluation

`make eval` builds fresh collections from [`sample_docs/`](sample_docs) (two fictional sets: an HR handbook with MD/PDF/DOCX files and a product manual with PDF/HTML/TXT files) and runs [`eval/questions.yaml`](eval/questions.yaml):

- **Retrieval hit rate:** the source containing the answer is among the retrieved chunks.
- **Answer correctness:** the expected facts appear in the answer.
- **Correct citation:** the answer cites the right source.
- **Out-of-scope refusal:** questions the documents can't answer, including questions about the *other* collection, are refused.

```
Retrieval hit rate @k : 21/21 (100%)
Answer correctness    : 21/21 (100%)
Correct citation      : 21/21 (100%)
Out-of-scope refusal  : 6/6 (100%)
Median latency        : 1.4s per question        (gpt-5.4-mini, Intel MacBook CPU)
```

`make eval ARGS=--no-llm` (or `uv run python -m eval.run_eval --no-llm`) measures retrieval only, without an API key.

The first eval run scored 20/21. A correct chunk was being dropped by a rerank-score threshold, and that miss led to the "the LLM decides not found" design above.

---

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)). `.env` is gitignored, so keys never reach the repo.

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `ollama` or `anthropic` |
| `LLM_MODEL` | `gpt-5.4-mini` | Any chat model of the chosen provider |
| `OPENAI_API_KEY` | – | Required for `openai` |
| `LLM_REASONING_EFFORT` | unset | `minimal`/`low`/`medium`/`high` for OpenAI reasoning models |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `fastembed` / `BAAI/bge-small-en-v1.5` | e.g. `BAAI/bge-m3` for multilingual, or `openai` / `text-embedding-3-small` |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | `RERANK_ENABLED=false` to turn it off |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K` | `400` / `60` / `5` | Defaults for new collections; overridable per collection |
| `DATA_DIR` | `./data` | SQLite, Chroma, uploads, model cache |
| `MAX_UPLOAD_MB` | `25` | Per file |

Each collection records the embedding model it was built with, so changing `EMBEDDING_MODEL` later only affects new collections and never mixes vector spaces.

---

## Project structure

```
app/
  main.py               FastAPI app factory, static UI routes
  config.py             Settings from env / .env
  service.py            RAGService: collections, ingestion, retrieval, answering (no web code)
  api/routes.py         REST + SSE endpoints
  ingestion/            parsers.py (5 formats → sections), chunker.py
  retrieval/            embedder.py, vector_store.py (Chroma), bm25.py, hybrid.py (RRF + rerank)
  generation/           llm.py (providers), prompts.py (grounding prompt, citations)
  storage/metadata.py   SQLite: collections, documents, ingestion status
  static/               index.html, chat.html, JS, CSS
tests/                  40 tests with fake embedder + fake LLM, no key or model download needed
eval/                   questions.yaml + run_eval.py
sample_docs/            Two fictional document sets
plan.md                 Original implementation plan
```

## Development

```bash
make dev     # auto-reload server
make test    # pytest
make eval    # quality evaluation (uses your LLM key)
make clean   # delete local indexes and uploads (keeps downloaded models)
```

**Intel Macs:** onnxruntime stopped publishing Intel-Mac wheels after 1.23, so `pyproject.toml` pins `onnxruntime<1.24` on that platform only.

## Limitations and next steps

- Scanned PDFs are reported as failed ("no extractable text"); OCR is not included.
- BM25 indexes are in memory per collection, and Chroma runs embedded. That suits thousands of documents, not millions; the next step would be a server vector DB with native hybrid search.
- No authentication or multi-tenant access control. Anyone who can reach the server can use every collection.
- Tables are flattened to text (HTML/DOCX rows are kept together as `cell | cell`).
- Possible additions: URL/website ingestion, answer feedback, and an LLM-as-judge faithfulness metric in the eval.

## How this was built

Built with Claude Code as an agentic coding exercise; the full session transcript is included with the submission. The plan is in [`plan.md`](plan.md), and each milestone is a separate commit.
