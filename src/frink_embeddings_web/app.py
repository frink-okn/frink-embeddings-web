import os

from flask import Flask
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


def create_app() -> Flask:
    app = Flask(__name__)

    # Configuration from environment with sensible defaults
    qdrant_host = os.getenv("QDRANT_HOST", "http://127.0.0.1")
    qdrant_port = int(os.getenv("QDRANT_PORT", "5554"))
    collection = os.getenv("QDRANT_COLLECTION", "OKN-Graph")
    model_name = os.getenv("SENTENCE_MODEL_NAME", "all-MiniLM-L6-v2")

    # Initialize shared dependencies once
    client = QdrantClient(qdrant_host, port=qdrant_port, timeout=30)
    app.config["QDRANT_CLIENT"] = client
    app.config["EMBEDDER"] = SentenceTransformer(model_name)
    app.config["QDRANT_COLLECTION"] = collection

    print(client)

    # Register API routes
    from frink_embeddings_web.routes import api

    app.register_blueprint(api)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run()
