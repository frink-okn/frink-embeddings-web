# Frink Embeddings Web Application

A Python application to serve HTTP resources to explore embeddings generated with [frink-okn/frink-embeddings](https://github.com/frink-okn/frink-embeddings).

Requirements
- Python 3.12+
- Qdrant running and accessible (default http://127.0.0.1:5554)

Configuration (env vars)
- QDRANT_LOCATION: default http://127.0.0.1:6663
- QDRANT_COLLECTION: default OKN-Graph
- QDRANT_HNSW_EF: default 500
- MODEL_NAME: default all-MiniLM-L6-v2
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
