from typing import TYPE_CHECKING

from cachetools import TTLCache, cached

if TYPE_CHECKING:
    from frink_embeddings_web.context import AppContext


@cached(
    cache=TTLCache(maxsize=1, ttl=60 * 10),
    key=lambda ctx: ctx.settings.qdrant_location

)
def get_graphs(ctx: "AppContext"):
    res = ctx.client.facet(
        collection_name=ctx.settings.qdrant_collection,
        key="graph",
        limit=100,
    )

    groups = [str(hit.value).strip() for hit in res.hits]
    return groups
