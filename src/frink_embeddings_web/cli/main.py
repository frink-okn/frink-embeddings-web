import json

import typer

from ..config import AppContext
from ..core.models import Query, TextFeature
from ..core.query import run_similarity_search


def search(
    term: str,
    graph: str | None = None,
    exact: bool = False,
    limit: int = 10,
):
    ctx = AppContext.from_env()

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

    for result in results.points:
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
