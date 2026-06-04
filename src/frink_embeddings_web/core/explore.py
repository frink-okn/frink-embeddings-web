import time
from dataclasses import dataclass

from loguru import logger
from qdrant_client.models import QueryRequest, ScoredPoint

from ..config import AppContext
from .graphs import get_graphs
from .models import Feature
from .query import build_graph_filter, get_embedding, make_search_params


@dataclass
class GraphSurveyResult:
    graph: str
    points: list[ScoredPoint]


def resolve_target_graphs(
    all_graphs: list[str],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[str]:
    """Resolve which graphs a survey should cover.

    `include` wins if given; otherwise all graphs minus `exclude`; otherwise
    every graph.
    """
    if include:
        return list(include)

    if exclude:
        excluded = set(exclude)
        return [g for g in all_graphs if g not in excluded]

    return list(all_graphs)


def _best_score(points: list[ScoredPoint]) -> float:
    # Qdrant returns each request's points best-first.
    if points and points[0].score is not None:
        return points[0].score
    return float("-inf")


def run_survey(
    ctx: AppContext,
    feature: Feature,
    *,
    include_graphs: list[str] | None = None,
    exclude_graphs: list[str] | None = None,
    limit: int = 5,
    exact: bool = False,
    hnsw_ef: int | None = None,
) -> list[GraphSurveyResult]:
    """Search each target graph independently for `feature`.

    Embeds the query once and issues one filtered similarity query per graph as
    a single batched request (Qdrant parallelizes them server-side). Returns one
    `GraphSurveyResult` per graph, sorted by best hit score descending.
    """
    all_graphs = get_graphs(ctx) if not include_graphs else []
    graphs = resolve_target_graphs(all_graphs, include_graphs, exclude_graphs)

    if not graphs:
        return []

    query_vector = get_embedding(ctx, feature).tolist()
    search_params = make_search_params(ctx, hnsw_ef=hnsw_ef, exact=exact)

    requests = [
        QueryRequest(
            query=query_vector,
            filter=build_graph_filter([graph], None),
            limit=limit,
            with_payload=True,
            params=search_params,
        )
        for graph in graphs
    ]

    start_time = time.perf_counter()
    responses = ctx.client.query_batch_points(
        collection_name=ctx.settings.qdrant_collection,
        requests=requests,
        timeout=ctx.settings.qdrant_timeout,
    )
    elapsed = time.perf_counter() - start_time

    logger.debug(f"{elapsed:.3f}s for survey over {len(graphs)} graphs")

    results = [
        GraphSurveyResult(graph=graph, points=resp.points)
        for graph, resp in zip(graphs, responses, strict=True)
    ]
    results.sort(key=lambda r: _best_score(r.points), reverse=True)

    return results
