import tomllib
from pathlib import Path

import typer
from loguru import logger
from pydantic import BaseModel
from qdrant_client.models import CollectionInfo

from ..config import AppContext, load_settings
from ..core.models import Query, TextFeature, TimedQueryResponse
from ..core.query import run_similarity_search

app = typer.Typer(pretty_exceptions_enable=False)


class EvalConfig(BaseModel):
    queries: dict[str, list[str]]

    @staticmethod
    def from_toml(path: Path):
        with path.open("rb") as fd:
            data = tomllib.load(fd)
        return EvalConfig.model_validate(data)


class QueryResult(BaseModel):
    target_graph: str
    query: str
    knn_results: TimedQueryResponse
    ann_ingraph_results: TimedQueryResponse
    ann_allgraph_results: TimedQueryResponse


class Evaluation(BaseModel):
    collection_settings: CollectionInfo
    hnsw_ef: int
    queries: list[QueryResult]


@app.command()
def run_eval(
    queries_toml: Path,
    output: Path,
):
    settings = load_settings()
    ctx = AppContext.from_settings(settings)
    config = EvalConfig.from_toml(queries_toml)

    limit = 50

    collection_settings = ctx.client.get_collection(
        ctx.settings.qdrant_collection
    )

    evaluation = Evaluation(
        collection_settings=collection_settings,
        hnsw_ef=settings.qdrant_hnsw_ef,
        queries=[],
    )

    for graph, terms in config.queries.items():
        for term in terms:
            logger.info(f"evaluating query '{term}' in graph '{graph}'")

            # Exact k-nearest neighbor search
            knn_results = run_similarity_search(
                ctx,
                query_obj=Query(
                    include_graphs=[graph],
                    feature=TextFeature(type="text", value=term),
                    limit=limit,
                ),
                exact=True,
            )

            # Approximate nearest neighbor search
            ann_results = run_similarity_search(
                ctx,
                query_obj=Query(
                    include_graphs=[graph],
                    feature=TextFeature(type="text", value=term),
                    limit=limit,
                ),
            )

            # Approximate nearest neighbor search without limiting to graph
            all_results = run_similarity_search(
                ctx,
                query_obj=Query(
                    feature=TextFeature(type="text", value=term),
                    limit=limit,
                ),
            )

            result = QueryResult(
                target_graph=graph,
                query=term,
                knn_results=knn_results,
                ann_ingraph_results=ann_results,
                ann_allgraph_results=all_results,
            )

            evaluation.queries.append(result)

            # recall, recall_overlap = recall_at_k(knn_results, ann_results)
            # recall_full, recall_full_verlap = recall_at_k(
            #     knn_results, all_results
            # )

            # all_results_min = all_results[-1].score
            # missing_points = [
            #     point
            #     for point in knn_results
            #     if point.score > all_results_min
            #     and point.id not in get_ids(all_results)
            # ]
            # print(f"recall@{limit} for in-graph ANN results: {recall}")
            # print(f"recall@{limit} for all-graph ANN results: {recall_full}")
            # print("Cosine similarity ranges:")
            # print(f"  KNN:      {fmt_range(knn_results)}")
            # print(f"  ANN:      {fmt_range(ann_results)}")
            # print(f"  Full ANN: {fmt_range(all_results)}")
            # print(f"missed points in full ANN: {len(missing_points)}")
            # print()

    with output.open("w") as fp:
        fp.write(evaluation.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
