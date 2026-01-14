import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    ScoredPoint,
    SearchParams,
)
from sentence_transformers import SentenceTransformer

from frink_embeddings_web.errors import URINotFoundError
from frink_embeddings_web.model import Feature, NodeFeature, Query, TextFeature


def embed_text(text: str, model: SentenceTransformer) -> np.ndarray:
    """Encode text into the same embedding space as the stored vectors."""
    return model.encode(text, normalize_embeddings=False).astype(np.float32)


def get_embedding(
    feature: Feature,
    client: QdrantClient,
    model: SentenceTransformer,
    collection_name: str,
) -> np.ndarray:
    match feature:
        case TextFeature(type="text"):
            return embed_text(feature.value, model)
        case NodeFeature(type="node"):
            points, _ = client.scroll(
                collection_name=collection_name,
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


def run_similarity_search(
    query_obj: Query,
    client: QdrantClient,
    model: SentenceTransformer,
    collection_name: str,
    hnsw_ef: int | None,
) -> list[ScoredPoint]:
    vector = get_embedding(query_obj.feature, client, model, collection_name)

    graph_filter: Filter | None = None

    if query_obj.include_graphs:
        graph_filter = Filter(
            must=[
                FieldCondition(
                    key="graph", match=MatchAny(any=query_obj.include_graphs)
                )
            ]
        )

    if query_obj.exclude_graphs:
        graph_filter = Filter(
            must_not=[
                FieldCondition(
                    key="graph", match=MatchAny(any=query_obj.exclude_graphs)
                )
            ]
        )

    search_params = SearchParams(hnsw_ef=hnsw_ef)

    return client.search(
        collection_name=collection_name,
        query_vector=vector.tolist(),
        query_filter=graph_filter,
        with_payload=True,
        limit=query_obj.limit,
        offset=query_obj.offset,
        search_params=search_params,
    )
