import numpy as np

from frink_embeddings_web.config.context import AppContext
from frink_embeddings_web.core.models import TextFeature
from frink_embeddings_web.core.query import get_embedding


def test_get_embedding_text_routes_through_embedder(ctx: AppContext):
    # The text branch embeds the query string through ctx.embedder.
    out = get_embedding(ctx, TextFeature(type="text", value="diabetes"))

    assert np.array_equal(out, ctx.embedder.embed("diabetes"))
