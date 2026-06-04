import time

import numpy as np
from loguru import logger
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    QuantizationSearchParams,
    SearchParams,
)

from ..config import AppContext
from .errors import URINotFoundError
from .models import (
    Feature,
    NodeFeature,
    Query,
    TextFeature,
    TimedQueryResponse,
)


def get_embedding(
    ctx: AppContext,
    feature: Feature,
) -> np.ndarray:
    match feature:
        case TextFeature(type="text"):
            return ctx.embedder.embed(feature.value)
        case NodeFeature(type="node"):
            points, _ = ctx.client.scroll(
                collection_name=ctx.settings.qdrant_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="iri", match=MatchValue(value=feature.value)
                        )
                    ]
                ),
                limit=1,
                with_vectors=True,
            )
            if not points:
                raise URINotFoundError(f"URI not found: {feature.value}")
            vec = points[0].vector
            return np.array(vec, dtype=np.float32)
        case _:
            raise ValueError("Unsupported feature type")


def build_graph_filter(
    include_graphs: list[str] | None,
    exclude_graphs: list[str] | None,
) -> Filter | None:
    """Build a Qdrant filter on the `graph` payload field.

    Include and exclude are mutually exclusive (enforced upstream by `Query`).
    """
    if include_graphs:
        return Filter(
            must=[
                FieldCondition(key="graph", match=MatchAny(any=include_graphs))
            ]
        )

    if exclude_graphs:
        return Filter(
            must_not=[
                FieldCondition(key="graph", match=MatchAny(any=exclude_graphs))
            ]
        )

    return None


def make_search_params(
    ctx: AppContext,
    hnsw_ef: int | None = None,
    exact: bool = False,
) -> SearchParams:
    return SearchParams(
        hnsw_ef=ctx.settings.qdrant_hnsw_ef if hnsw_ef is None else hnsw_ef,
        exact=exact,
        quantization=QuantizationSearchParams(
            ignore=False,
            rescore=True,
            oversampling=3.0,
        ),
    )


def run_similarity_search(
    ctx: AppContext,
    query_obj: Query,
    hnsw_ef: int | None = None,
    exact: bool = False,
) -> TimedQueryResponse:
    vector = get_embedding(ctx, query_obj.feature)

    graph_filter = build_graph_filter(
        query_obj.include_graphs, query_obj.exclude_graphs
    )
    search_params = make_search_params(ctx, hnsw_ef=hnsw_ef, exact=exact)

    start_time = time.perf_counter()
    resp = ctx.client.query_points(
        query=vector.tolist(),
        collection_name=ctx.settings.qdrant_collection,
        query_filter=graph_filter,
        with_payload=True,
        limit=query_obj.limit,
        offset=query_obj.offset,
        search_params=search_params,
        timeout=ctx.settings.qdrant_timeout,
    )
    end_time = time.perf_counter()

    query_time = end_time - start_time

    logger.debug(f"{query_time:.3f}s for query: {query_obj}")

    return TimedQueryResponse(
        points=resp.points,
        time=end_time - start_time,
    )
