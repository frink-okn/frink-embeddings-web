import numpy as np

from okn_embeddings.config.context import AppContext
from okn_embeddings.core.models import TextFeature
from okn_embeddings.core.query import get_embedding


def test_get_embedding_text_routes_through_embedder(ctx: AppContext):
    # The text branch embeds the query string through ctx.embedder.
    out = get_embedding(ctx, TextFeature(type="text", value="diabetes"))

    assert np.array_equal(out, ctx.embedder.embed("diabetes"))
