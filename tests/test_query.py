import numpy as np

from frink_embeddings_web.core.models import TextFeature
from frink_embeddings_web.core.query import get_embedding


class _StubEmbedder:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return self.vector


class _Ctx:
    def __init__(self, embedder):
        self.embedder = embedder


def test_get_embedding_text_routes_through_embedder():
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    embedder = _StubEmbedder(vector)
    ctx = _Ctx(embedder)

    out = get_embedding(ctx, TextFeature(type="text", value="diabetes"))

    assert out is vector
    assert embedder.calls == ["diabetes"]
