import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from frink_embeddings_web.config.context import AppContext
from frink_embeddings_web.config.settings import AppSettings
from frink_embeddings_web.core.embedding import FastEmbedEmbedder

_COLLECTION = "test-graph"
_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def embedder() -> FastEmbedEmbedder:
    # The real embedder, loaded once for the whole test session.
    return FastEmbedEmbedder(_MODEL)


@pytest.fixture
def ctx(embedder: FastEmbedEmbedder) -> AppContext:
    # A real AppContext over an in-memory Qdrant: no server, a fresh DB per
    # test, but the actual client / embedder / query path -- no fakes.
    settings = AppSettings(
        qdrant_location=":memory:",
        qdrant_collection=_COLLECTION,
        qdrant_hnsw_ef=128,
        qdrant_timeout=30,
        model_name=_MODEL,
    )
    client = QdrantClient(":memory:")
    dim = len(embedder.embed("dimension probe"))
    client.create_collection(
        _COLLECTION, VectorParams(size=dim, distance=Distance.COSINE)
    )
    return AppContext(client=client, embedder=embedder, settings=settings)
