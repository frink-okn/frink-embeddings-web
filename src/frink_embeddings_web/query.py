import numpy as np
from qdrant_client import QdrantClient, Filter, FieldCondition, MatchValue
from frink_embeddings_web.model import Query, Feature, TextFeature, NodeFeature

QDRANT_URI = "http://127.0.0.1/"
QDRANT_PORT = 5554
QDRANT_COLLECTION_NAME = "OKN-Graph"

def embed_text(text: str):
    ...

def get_embedding(feature: Feature) -> list[float]:
    match feature:
        case TextFeature(type="text"):
            return embed_text(feature.value)
        case NodeFeature(type="node"):
            res = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(key="iri", match=MatchValue(value=feature.value))]
                ),
                limit=1,
            )
            return np.array(res[0][0].vector)
            # Get the vector from the QDrant client
        case _:
            raise ValueError()

def query(query: Query, limit: int=10):
    positive_vectors = [get_embedding(feature) for feature in query.positive]
    negative_vectors = [get_embedding(feature) for feature in query.negative]

    query_vector = np.mean(positive_vectors, axis=0)

    if negative_vectors:
        negative_mean = np.mean(negative_vectors, axis=0)
        query_vector = query_vector - negative_mean

    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    return client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=limit
    )



def main():
    client = QdrantClient(QDRANT_URI, port=QDRANT_PORT)
    collection = client.get_collection(QDRANT_COLLECTION_NAME)

    item = client.retrieve(
        QDRANT_COLLECTION_NAME,
        ["http://example.com/"],
    )

if __name__ == "__main__":
    main()
