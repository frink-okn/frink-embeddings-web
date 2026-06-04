import numpy as np
import pytest
from pydantic import ValidationError
from qdrant_client.models import Filter, ScoredPoint

from frink_embeddings_web.core.explore import (
    resolve_target_graphs,
    run_survey,
)
from frink_embeddings_web.core.models import (
    NodeFeature,
    TextFeature,
    build_feature,
)
from frink_embeddings_web.core.query import build_graph_filter


def test_resolve_target_graphs_include_wins():
    assert resolve_target_graphs(["a", "b", "c"], ["b", "c"], None) == [
        "b",
        "c",
    ]


def test_resolve_target_graphs_exclude():
    assert resolve_target_graphs(["a", "b", "c"], None, ["b"]) == ["a", "c"]


def test_resolve_target_graphs_all():
    assert resolve_target_graphs(["a", "b"], None, None) == ["a", "b"]


def test_build_graph_filter_include():
    f = build_graph_filter(["a", "b"], None)
    assert isinstance(f, Filter)
    assert f.must is not None
    assert f.must_not is None


def test_build_graph_filter_exclude():
    f = build_graph_filter(None, ["a"])
    assert isinstance(f, Filter)
    assert f.must_not is not None
    assert f.must is None


def test_build_graph_filter_none():
    assert build_graph_filter(None, None) is None


def test_build_feature_text():
    feat = build_feature("text", "hello")
    assert isinstance(feat, TextFeature)
    assert feat.value == "hello"


def test_build_feature_node():
    assert isinstance(build_feature("node", "urn:x"), NodeFeature)


def test_build_feature_bad_type():
    with pytest.raises(ValidationError):
        build_feature("bogus", "x")


# --- run_survey against a stub client (no network) ---


class _FakeResponse:
    def __init__(self, points):
        self.points = points


class _FakeSettings:
    qdrant_collection = "C"
    qdrant_timeout = 5
    qdrant_hnsw_ef = 128


class _FakeClient:
    def __init__(self, points_by_graph):
        self.points_by_graph = points_by_graph
        self.received_requests = None

    def query_batch_points(self, collection_name, requests, timeout=None):
        self.received_requests = requests
        # Respond in request order; map each request's graph filter to its
        # canned points so we exercise the zip-back-to-graphs logic.
        responses = []
        for req in requests:
            graph = req.filter.must[0].match.any[0]
            responses.append(_FakeResponse(self.points_by_graph.get(graph, [])))
        return responses


class _FakeCtx:
    def __init__(self, client):
        self.client = client
        self.settings = _FakeSettings()
        self.model = None


def _point(score):
    return ScoredPoint(id=1, version=0, score=score, payload={})


def test_run_survey_orders_by_best_score_and_embeds_once(monkeypatch):
    calls = {"embed": 0}

    def fake_embed(ctx, feature):
        calls["embed"] += 1
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)

    monkeypatch.setattr(
        "frink_embeddings_web.core.explore.get_embedding", fake_embed
    )

    client = _FakeClient(
        {
            "g_low": [_point(0.2), _point(0.1)],
            "g_high": [_point(0.9), _point(0.8)],
            "g_mid": [_point(0.5)],
        }
    )
    ctx = _FakeCtx(client)

    results = run_survey(
        ctx,
        build_feature("text", "q"),
        include_graphs=["g_low", "g_high", "g_mid"],
        limit=2,
    )

    # Embedded exactly once, reused for every graph.
    assert calls["embed"] == 1
    # One batched request per target graph, with the per-graph limit applied.
    assert len(client.received_requests) == 3
    assert all(req.limit == 2 for req in client.received_requests)
    # Results sorted by best hit score, descending.
    assert [r.graph for r in results] == ["g_high", "g_mid", "g_low"]
    assert results[0].points[0].score == 0.9


def test_run_survey_all_graphs_uses_get_graphs(monkeypatch):
    monkeypatch.setattr(
        "frink_embeddings_web.core.explore.get_embedding",
        lambda ctx, feature: np.array([0.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "frink_embeddings_web.core.explore.get_graphs",
        lambda ctx: ["a", "b"],
    )

    client = _FakeClient({"a": [_point(0.3)], "b": [_point(0.7)]})
    results = run_survey(_FakeCtx(client), build_feature("text", "q"))

    assert {r.graph for r in results} == {"a", "b"}
    assert [r.graph for r in results] == ["b", "a"]
