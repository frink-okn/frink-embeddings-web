from flask import Blueprint, request, jsonify, render_template
from pydantic import ValidationError
from qdrant_client.models import ScoredPoint

from frink_embeddings_web.context import get_ctx
from frink_embeddings_web.model import Query
from frink_embeddings_web.query import run_similarity_search

api = Blueprint("api", __name__)
web = Blueprint("web", __name__)

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

    ctx = get_ctx()

    try:
        points = run_similarity_search(
            query_obj=q,
            client=ctx.client,
            model=ctx.model,
            collection_name=ctx.collection,
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

@web.get("/")
def index():
    return render_template("index.html")

@web.get("/feature-row")
def feature_row():
    return render_template("partials/feature_row.html")

@web.get("/noop")
def noop():
    return ""

@web.post("/query-view")
def post_query_view():
    form = request.form
    types = form.getlist("feat_type[]")
    values = form.getlist("feat_value[]")
    signs = form.getlist("feat_sign[]")

    positives = []
    negatives = []

    for t, v, s in zip(types, values, signs):
        t_norm = "text" if str(t).lower().startswith("text") else "node"
        feature = {"type": t_norm, "value": v}
        if str(s).lower().startswith("pos"):
            positives.append(feature)
        else:
            negatives.append(feature)

    if not positives:
        return render_template("partials/results_table.html", results=[], error="At least one positive feature is required."), 400

    data = {
        "positive": positives,
        "negative": negatives,
        "graphs": None,
    }

    try:
        q = Query.model_validate(data)
    except ValidationError:
        return render_template("partials/results_table.html", results=[], error="Invalid request."), 400

    ctx = get_ctx()
    try:
        points = run_similarity_search(
            query_obj=q,
            client=ctx.client,
            model=ctx.model,
            collection_name=ctx.collection,
            limit=int(form.get("limit", 10)),
        )
    except ValueError as e:
        msg = str(e)
        status = 404 if msg.startswith("IRI not found") else 400
        return render_template("partials/results_table.html", results=[], error=msg), status
    except Exception as e:
        return render_template("partials/results_table.html", results=[], error=f"internal error: {e}"), 500

    results = [serialize_point(p) for p in points]
    return render_template("partials/results_table.html", results=results)
