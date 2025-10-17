import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from frink_embeddings_web.context import AppContext, WrappedFlask


def create_app() -> WrappedFlask:
    app = WrappedFlask(__name__)

    # Configuration from environment with sensible defaults
    qdrant_host = os.getenv("QDRANT_HOST", "http://127.0.0.1")
    qdrant_port = int(os.getenv("QDRANT_PORT", "5554"))
    collection = os.getenv("QDRANT_COLLECTION", "OKN-Graph")
    model_name = os.getenv("SENTENCE_MODEL_NAME", "all-MiniLM-L6-v2")

    # Initialize shared dependencies once
    client = QdrantClient(qdrant_host, port=qdrant_port, timeout=30)
    model = SentenceTransformer(model_name)

    ctx = AppContext(
        client=client,
        model=model,
        collection=collection,
    )
    app.ctx = ctx

    # Register API & Web routes
    from frink_embeddings_web.routes import api, web

    app.register_blueprint(api)
    app.register_blueprint(web)

    return app


app = create_app()


if __name__ == "__main__":
    app.run()
