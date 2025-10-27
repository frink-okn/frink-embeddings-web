from flask import Blueprint, jsonify, render_template, request
from pydantic import ValidationError
from qdrant_client.models import ScoredPoint

from frink_embeddings_web.context import get_ctx
from frink_embeddings_web.errors import URINotFoundError
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
    except Exception as e:
        match e:
            case URINotFoundError():
                status = 404
            case ValueError():
                status = 400
            case _:
                status = 500
        return jsonify({"error": str(e)}), status

    return jsonify({"results": [serialize_point(p) for p in points]})


@web.get("/")
def index():
    feature_type = request.args.get("type", "Text")
    feature_value = request.args.get("value", "")
    return render_template(
        "index.html",
        feature_type=feature_type,
        feature_value=feature_value,
    )


@web.post("/query-view")
def post_query_view():
    form = request.form

    data = {
        "feature": {
            "type": form.get("feat_type"),
            "value": form.get("feat_value"),
        },
        "graphs": form.getlist("graphs[]"),
    }

    try:
        q = Query.model_validate(data)
    except ValidationError:
        return render_template(
            "partials/results_table.html",
            results=[],
            error="Invalid query.",
        ), 400

    ctx = get_ctx()
    try:
        points = run_similarity_search(
            query_obj=q,
            client=ctx.client,
            model=ctx.model,
            collection_name=ctx.collection,
            limit=int(form.get("limit", 10)),
        )
    except Exception as e:
        match e:
            case URINotFoundError():
                status = 404
            case ValueError():
                status = 400
            case _:
                status = 500
        return render_template(
            "partials/results_table.html",
            results=[],
            error=str(e),
        ), status

    results = [serialize_point(p) for p in points]
    return render_template("partials/results_table.html", results=results)
