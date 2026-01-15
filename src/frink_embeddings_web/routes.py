from urllib.parse import quote

import httpx
from flask import Blueprint, jsonify, render_template, request
from pydantic import ValidationError
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import ScoredPoint

from frink_embeddings_web.context import get_ctx
from frink_embeddings_web.errors import URINotFoundError
from frink_embeddings_web.graphs import update_graph_catalog
from frink_embeddings_web.model import Query
from frink_embeddings_web.query import run_similarity_search

api = Blueprint("api", __name__)
web = Blueprint("web", __name__)


def serialize_point(p: ScoredPoint) -> dict:
    payload = p.payload or {}

    return {
        "id": str(p.id),
        "score": float(p.score) if p.score is not None else None,
        "payload": p.payload or {},
        "encoded_uri": quote(payload.get("iri", ""), safe=""),
    }


def unwrap_qdrant_error(e: Exception) -> Exception:
    is_qdrant_wrapped = (
        isinstance(e, ResponseHandlingException)
        and e.args
        and isinstance(e.args[0], Exception)
    )

    return e.args[0] if is_qdrant_wrapped else e


def parse_error(e: Exception):
    inner = unwrap_qdrant_error(e)
    msg = str(inner)
    match inner:
        case URINotFoundError():
            status = 404
        case httpx.ConnectError():
            status = 500
            msg = "Could not connect to Qdrant server"
        case ValueError():
            status = 400
        case _:
            status = 500
    return msg, status


@api.post("/update-graphs")
def update_graphs():
    ctx = get_ctx()
    update_graph_catalog(ctx)
    return "", 200


@api.post("/query")
def post_query():
    data = request.get_json(silent=True) or {}

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
            collection_name=ctx.settings.qdrant_collection,
            hnsw_ef=ctx.settings.qdrant_hnsw_ef,
        )
    except Exception as e:
        msg, status = parse_error(e)
        return jsonify({"error": msg}), status

    return jsonify({"results": [serialize_point(p) for p in points]})


@web.get("/")
def index():
    ctx = get_ctx()

    feature_type = request.args.get("type", "Text")
    feature_value = request.args.get("value", "")
    graphs = ctx.graphs

    graph_mode = request.args.get("graph-mode", "include")
    selected_graphs = request.args.getlist("graph")

    return render_template(
        "index.html",
        feature_type=feature_type,
        feature_value=feature_value,
        graphs=graphs,
        graph_mode=graph_mode,
        selected_graphs=selected_graphs,
    )


@web.post("/query-view")
def post_query_view():
    form = request.form

    include_graphs = form.getlist("include_graphs")
    exclude_graphs = form.getlist("exclude_graphs")

    data = {
        "feature": {
            "type": form.get("feat_type"),
            "value": form.get("feat_value"),
        },
        "limit": form.get("limit", 10),
        "offset": form.get("offset", 0),
    }

    if include_graphs:
        data["include_graphs"] = include_graphs

    if exclude_graphs:
        data["exclude_graphs"] = exclude_graphs

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
            collection_name=ctx.settings.qdrant_collection,
            hnsw_ef=ctx.settings.qdrant_hnsw_ef,
        )
    except Exception as e:
        msg, status = parse_error(e)
        return render_template(
            "partials/results_table.html",
            results=[],
            error=msg,
        ), status

    results = [serialize_point(p) for p in points]
    return render_template(
        "partials/results_table.html",
        results=results,
        query=q,
    )
