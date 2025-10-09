# Frink Embeddings Web Application

A Python application to serve HTTP resources to explore embeddings generated with [frink-okn/frink-embeddings](https://github.com/frink-okn/frink-embeddings).

Requirements
- Python 3.12+
- Qdrant running and accessible (default http://127.0.0.1:5554)

Configuration (env vars)
- QDRANT_HOST: default http://127.0.0.1
- QDRANT_PORT: default 5554
- QDRANT_COLLECTION: default OKN-Graph
- SENTENCE_MODEL_NAME: default all-MiniLM-L6-v2
- HOST: HTTP server bind host (default 0.0.0.0)
- PORT: HTTP server bind port (default 8000)
- DEBUG: Set to "1" to enable Flask debug


