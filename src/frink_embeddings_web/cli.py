import json

import typer

from frink_embeddings_web.context import AppContext
from frink_embeddings_web.model import Query, TextFeature
from frink_embeddings_web.query import run_similarity_search
from frink_embeddings_web.settings import load_settings


def search(
    term: str,
    graph: str | None = None,
    exact: bool = False,
    limit: int = 10,
):
    settings = load_settings()
    settings.qdrant_timeout = 60
    ctx = AppContext.from_settings(settings)

    query = Query(
        include_graphs=[graph] if graph else None,
        feature=TextFeature(type="text", value=term),
        limit=limit,
    )

    results = run_similarity_search(
        ctx,
        query_obj=query,
        exact=exact,
    )

    out = []

    for result in results:
        if not result.payload:
            continue
        out.append(
            {
                "score": result.score,
                "payload": result.payload,
            }
        )

    print(json.dumps(out))


if __name__ == "__main__":
    typer.run(search)
