import tomllib
from pathlib import Path

import typer
from pydantic import BaseModel
from qdrant_client.models import ScoredPoint

from frink_embeddings_web.context import AppContext
from frink_embeddings_web.model import Query, TextFeature
from frink_embeddings_web.query import run_similarity_search
from frink_embeddings_web.settings import load_settings


class EvalConfig(BaseModel):
    queries: dict[str, list[str]]

    @staticmethod
    def from_toml(path: Path):
        with path.open("rb") as fd:
            data = tomllib.load(fd)
        return EvalConfig.model_validate(data)


def get_ids(results: list[ScoredPoint]):
    ids: list[str] = []
    for r in results:
        if r.payload and "iri" in r.payload:
            ids.append(r.payload["iri"])
    return ids


def recall_at_k(a: list[ScoredPoint], b: list[ScoredPoint]):
    a_ids = set(get_ids(a))
    b_ids = set(get_ids(b))
    overlap = len(a_ids & b_ids)
    return overlap / len(a), overlap


def fmt_range(points: list[ScoredPoint]):
    return f"{points[-1].score:.2f} to {points[0].score:.2f}"


def run_eval(
    queries_toml: Path,
):
    settings = load_settings()
    settings.qdrant_timeout = 60
    ctx = AppContext.from_settings(settings)
    config = EvalConfig.from_toml(queries_toml)

    limit = 50

    for graph, terms in config.queries.items():
        for term in terms:
            print(f"evaluating query '{term}' in graph '{graph}'")
            # Exact k-nearest neighbor search
            knn_results = run_similarity_search(
                query_obj=Query(
                    include_graphs=[graph],
                    feature=TextFeature(type="text", value=term),
                    limit=limit,
                ),
                client=ctx.client,
                model=ctx.model,
                collection_name=settings.qdrant_collection,
                exact=True,
                hnsw_ef=settings.qdrant_hnsw_ef,
            )

            # Approximate nearest neighbor search
            ann_results = run_similarity_search(
                query_obj=Query(
                    include_graphs=[graph],
                    feature=TextFeature(type="text", value=term),
                    limit=limit,
                ),
                client=ctx.client,
                model=ctx.model,
                collection_name=settings.qdrant_collection,
                exact=False,
                hnsw_ef=settings.qdrant_hnsw_ef,
            )

            # Approximate nearest neighbor search without limiting to graph
            all_results = run_similarity_search(
                query_obj=Query(
                    feature=TextFeature(type="text", value=term),
                    limit=limit,
                ),
                client=ctx.client,
                model=ctx.model,
                collection_name=settings.qdrant_collection,
                exact=False,
                hnsw_ef=settings.qdrant_hnsw_ef,
            )

            recall, recall_overlap = recall_at_k(knn_results, ann_results)
            recall_full, recall_full_verlap = recall_at_k(
                knn_results, all_results
            )

            all_results_min = all_results[-1].score
            missing_points = [
                point for point in knn_results
                if point.score > all_results_min and
                point.id not in get_ids(all_results)
            ]
            print(f"recall@{limit} for in-graph ANN results: {recall}")
            print(f"recall@{limit} for all-graph ANN results: {recall_full}")
            print("Cosine similarity ranges:")
            print(f"  KNN:      {fmt_range(knn_results)}")
            print(f"  ANN:      {fmt_range(ann_results)}")
            print(f"  Full ANN: {fmt_range(all_results)}")
            print(f"missed points in full ANN: {len(missing_points)}")
            print()


if __name__ == "__main__":
    typer.run(run_eval)
