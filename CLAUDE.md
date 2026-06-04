# frink-embeddings-web

Python app to **produce** and **search** vector embeddings of RDF knowledge graphs. The
indexing pipeline turns graph nodes into text and embeds them; the search interfaces embed a
query (text or an existing node) with the same embedding model (fastembed / ONNX, via
`core/embedding.py`) and match it against a Qdrant collection via similarity search, with
optional filtering by source graph. (Embedding upload to Qdrant lives on a separate branch.)

## Architecture

One shared search core powers three interfaces:

- **Core** — `core/query.py::run_similarity_search` is the single entry point for search,
  driven by the `Query` / `TextFeature` / `NodeFeature` models in `core/models.py`. It embeds
  the feature (`embed_text` or by looking up a node's stored vector), builds an
  include/exclude `graph` filter, and calls Qdrant `query_points`.
- **HTTP JSON API** — `web/routes.py`: `POST /query` accepts a `Query` JSON body and returns
  serialized scored points.
- **HTMX web UI** — `web/routes.py`: `GET /` renders the form; `POST /query-view` returns an
  HTML results partial. Templates in `web/templates/`, assets in `web/static/`.
- **CLI** — `cli/main.py`: the `frink-search` Typer app. `search` (text or node/IRI query,
  `--graph`/`--exclude-graph`, `--limit`/`--offset`, `--exact`, `--show-repr`, `--json`),
  `survey` (per-graph top-N across every graph or a `-g`/`-x` subset — `core/explore.py`,
  one Qdrant batch request) and `list-graphs` (point counts, `--sort name|count`).

Supporting modules:

- **indexing/** — Typer app (`frink-indexing`) that produces the embeddings: textify/materialize
  RDF (HDT) graphs into embedding records and sample types/targets. See also the `textify`
  skill. (Uploading the resulting embeddings to Qdrant lives on a separate branch.)
- **evaluation/** — compares kNN vs ANN search quality and renders a markdown report.
- **config/** — `AppSettings` (pydantic-settings) loaded from `default.env` then `.env`;
  `AppContext.from_env()` constructs the Qdrant client and embedding model.

## Requirements

- Python 3.12+
- A running Qdrant instance (default `http://127.0.0.1:6663`)

## Configuration

Env vars (defaults in `default.env`, overridable via `.env`):

- `QDRANT_LOCATION`, `QDRANT_COLLECTION` (`OKN-Graph`), `QDRANT_HNSW_EF`, `QDRANT_TIMEOUT`
- `MODEL_NAME` (`sentence-transformers/all-MiniLM-L6-v2`)
- Web server: `HOST`, `PORT`, `NUM_WORKERS`, `DEBUG`, `SCRIPT_NAME` (subdir hosting, gunicorn)

## Common commands

Use `uv` for everything.

- `make dev` — run the Flask dev server
- `make dev-gunicorn` — run under gunicorn
- `make test` — `uv run pytest`
- `make lint` — `ruff check` + `ruff format --check`
- `make format` — `ruff check --fix` + `ruff format`
- `make docker-build` / `make docker-run`
- `uv run frink-indexing --help` — indexing/materialization CLI
- `uv run frink-search --help` — search CLI (`search`, `list-graphs`)

## Conventions

- Style enforced by ruff: line length 80, target py312, rule set B/E/F/I/S/W. Run
  `make format` before committing.
- Do not add `from __future__ import annotations`; native 3.12 typing is fine.
- All three interfaces should go through `run_similarity_search` — add features to the core
  and the `Query` model rather than duplicating search logic per interface. Shared helpers
  `build_feature`/`build_query` (`core/models.py`), `build_graph_filter`/`make_search_params`
  (`core/query.py`) and `summarize_point` (`core/results.py`) keep request assembly and result
  formatting in one place; `core/explore.py::run_survey` returns render-agnostic data for a
  future web caller.
