from flask import Blueprint, request, jsonify, current_app
from pydantic import ValidationError
from qdrant_client.models import ScoredPoint

from frink_embeddings_web.model import Query
from frink_embeddings_web.query import run_similarity_search

api = Blueprint("api", __name__)

def serialize_point(p: ScoredPoint) -> dict:
    return {
        "id": str(p.id),
        "score": float(p.score) if p.score is not None else None,
        "payload": p.payload or {},
    }

@api.post("/query")
def post_query():
    data = request.get_json(silent=True) or {}

    # Allow missing negatives by defaulting to empty list
    if "negative" not in data:
        data["negative"] = []

    # Require at least one positive feature
    if not data.get("positive"):
        return jsonify({"error": "positive features required"}), 400

    limit = int(data.get("limit", 10))

    try:
        q = Query.model_validate(data)
    except ValidationError as e:
        return jsonify({"error": "invalid request", "details": e.errors()}), 400

    client = current_app.config["QDRANT_CLIENT"]
    model = current_app.config["EMBEDDER"]
    collection = current_app.config["QDRANT_COLLECTION"]

    try:
        points = run_similarity_search(
            query_obj=q,
            client=client,
            model=model,
            collection_name=collection,
            limit=limit,
        )
    except ValueError as e:
        # Return 404 on missing IRI, else 400 for other ValueErrors
        msg = str(e)
        if msg.startswith("IRI not found"):
            return jsonify({"error": msg}), 404
        return jsonify({"error": msg}), 400
    except Exception as e:
        return jsonify({"error": "internal error", "message": str(e)}), 500

    return jsonify({"results": [serialize_point(p) for p in points]})
