from dataclasses import dataclass
from typing import TYPE_CHECKING

from cachetools import TTLCache, cached

if TYPE_CHECKING:
    from ..config import AppContext


@dataclass(frozen=True)
class GraphFacet:
    graph: str
    count: int


@cached(
    cache=TTLCache(maxsize=1, ttl=60 * 10),
    key=lambda ctx: ctx.settings.qdrant_location,
)
def get_graph_facets(ctx: "AppContext") -> list[GraphFacet]:
    res = ctx.client.facet(
        collection_name=ctx.settings.qdrant_collection,
        key="graph",
        limit=100,
    )

    return [GraphFacet(str(hit.value).strip(), hit.count) for hit in res.hits]


def get_graphs(ctx: "AppContext") -> list[str]:
    return [facet.graph for facet in get_graph_facets(ctx)]
