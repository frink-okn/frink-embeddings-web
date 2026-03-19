from pathlib import Path

import numpy as np
import pandas as pd
import typer
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Sample,
    SampleQuery,
)

from ..config import AppContext, load_settings


def sample_graph_points(
    limit: int = 1_000,
    output_dir: Path = Path("samples"),
):
    """
    For each graph collection, sample N points, and write them to individual
    parquet files.
    """
    settings = load_settings()
    ctx = AppContext.from_settings(settings)

    graph_facet = ctx.client.facet(
        collection_name=ctx.settings.qdrant_collection,
        key="graph",
        limit=100,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for hit in graph_facet.hits:
        graph = hit.value
        print(f"querying sample for graph {graph}")

        sample = ctx.client.query_points(
            collection_name=ctx.settings.qdrant_collection,
            query=SampleQuery(sample=Sample.RANDOM),
            query_filter=Filter(
                must=[
                    FieldCondition(key="graph", match=MatchValue(value=graph))
                ]
            ),
            with_payload=True,
            with_vectors=True,
            limit=limit,
        )

        df = pd.DataFrame(
            [
                {
                    "point_id": str(point.id),
                    "label": point.payload["label"],
                    "iri": point.payload["iri"],
                    "repr": point.payload["repr"],
                    "graph": graph,
                    "vector": np.asarray(
                        point.vector, dtype=np.float32
                    ).tolist(),
                }
                for point in sample.points
                if point.vector
            ]
        )
        print(f"got {len(df)} sampled points")

        output_path = output_dir / f"{graph}.parquet"
        df.to_parquet(output_path, index=False, compression="zstd")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    typer.run(sample_graph_points)
