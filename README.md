# Frink Embeddings Web Application

A Python application to serve HTTP resources to explore embeddings generated with [frink-okn/frink-embeddings](https://github.com/frink-okn/frink-embeddings).

Requirements
- Python 3.12+
- Qdrant running and accessible (default http://127.0.0.1:5554)

Configuration (env vars)
- QDRANT_LOCATION: default http://127.0.0.1:6663
- QDRANT_COLLECTION: default OKN-Graph
- QDRANT_HNSW_EF: default 500
- MODEL_NAME: default sentence-transformers/all-MiniLM-L6-v2
- HOST: HTTP server bind host (default 0.0.0.0)
- PORT: HTTP server bind port (default 8000)
- NUM_WORKERS: Number of gunicorn workers to use (default 4)
- DEBUG: Set to "1" to enable Flask debug

To host under a subdirectory (gunicorn only), set the SCRIPT_NAME environment variable.

# Running

To run a local server using Flask, run:

```
make dev
```

# Command-line search

The `frink-search` CLI queries the same embeddings as the web app (it reads the same
`QDRANT_*` / `MODEL_NAME` configuration):

```
# Similarity search by text (table output)
uv run frink-search search "diabetes" --limit 5

# Search by an existing node's IRI ("find similar")
uv run frink-search search https://example.org/node/123 --type node

# Restrict to (or exclude) specific graphs; JSON output for scripting
uv run frink-search search "diabetes" -g GraphA -g GraphB --json
uv run frink-search search "diabetes" -x GraphA

# Survey: top matches *per graph* across every graph (or a -g/-x subset)
uv run frink-search survey "diabetes" --limit 3
uv run frink-search survey "diabetes" -g GraphA -g GraphB

# List the graphs in the collection and their point counts
uv run frink-search list-graphs --sort count
```

Run `uv run frink-search --help` (or `... search --help`) for all options.

# Building a docker image

To create a Docker image that will run the server using gunicorn, run:

```
make docker-build
```

To test your image, run:

```
make docker-run
```

By default, this image is called `frink-web`. Change the name by setting the
environment variable `DOCKER_NAME`.
