import pytest
from pydantic import ValidationError

from frink_embeddings_web.core.models import (
    NodeFeature,
    TextFeature,
    build_query,
)


def test_build_query_text_defaults():
    q = build_query(feature_type="text", value="diabetes")

    assert isinstance(q.feature, TextFeature)
    assert q.feature.value == "diabetes"
    assert q.limit == 10
    assert q.offset == 0
    assert q.include_graphs is None
    assert q.exclude_graphs is None


def test_build_query_node_feature():
    q = build_query(feature_type="node", value="http://example.com/x")

    assert isinstance(q.feature, NodeFeature)
    assert q.feature.value == "http://example.com/x"


def test_build_query_coerces_string_limits():
    q = build_query(feature_type="text", value="x", limit="25", offset="50")

    assert q.limit == 25
    assert q.offset == 50


def test_build_query_include_graphs():
    q = build_query(
        feature_type="text",
        value="x",
        include_graphs=["GraphA", "GraphB"],
    )

    assert q.include_graphs == ["GraphA", "GraphB"]
    assert q.exclude_graphs is None


def test_build_query_empty_graph_lists_become_none():
    q = build_query(
        feature_type="text",
        value="x",
        include_graphs=[],
        exclude_graphs=[],
    )

    assert q.include_graphs is None
    assert q.exclude_graphs is None


def test_build_query_rejects_both_include_and_exclude():
    with pytest.raises(ValidationError):
        build_query(
            feature_type="text",
            value="x",
            include_graphs=["GraphA"],
            exclude_graphs=["GraphB"],
        )


def test_build_query_rejects_unknown_feature_type():
    with pytest.raises(ValidationError):
        build_query(feature_type="bogus", value="x")
