import json

import numpy as np
import pytest

from frink_embeddings_web.core.embedding import FastEmbedEmbedder
from frink_embeddings_web.core.results import summarize_point
from frink_embeddings_web.indexing.upload import (
    chunks,
    iter_jsonl,
    payload_for_record,
    point_id,
    upload_file,
)

# --- point_id: deterministic and idempotent ---


def test_point_id_is_deterministic():
    rec = {"iris": ["urn:a"], "embedding_text": "hello"}
    assert point_id("g", rec) == point_id("g", rec)


def test_point_id_varies_by_graph_iri_and_text():
    base = {"iris": ["urn:a"], "embedding_text": "hello"}
    pid = point_id("g", base)
    assert pid != point_id("other", base)
    assert pid != point_id("g", {"iris": ["urn:b"], "embedding_text": "hello"})
    assert pid != point_id("g", {"iris": ["urn:a"], "embedding_text": "bye"})


def test_point_id_accepts_singular_iri_schema():
    # materialize.py emits a singular `iri`; index.py emits an `iris` list.
    singular = point_id("g", {"iri": "urn:a", "embedding_text": "hi"})
    plural = point_id("g", {"iris": ["urn:a"], "embedding_text": "hi"})
    assert singular == plural


# --- payload_for_record: matches what summarize_point reads ---


def test_payload_maps_to_query_side_fields():
    record = {
        "iris": ["urn:a", "urn:b"],
        "label": "A Thing",
        "embedding_text": "label: A Thing\ntype: Foo",
    }
    payload = payload_for_record("my-graph", record)

    assert payload["graph"] == "my-graph"
    assert payload["iri"] == ["urn:a", "urn:b"]
    assert payload["repr"] == "label: A Thing\ntype: Foo"
    assert payload["label"] == "A Thing"
    # The renamed source keys are not carried through verbatim.
    assert "embedding_text" not in payload
    assert "iris" not in payload


def test_payload_round_trips_through_summarize_point():
    record = {
        "iris": ["urn:a", "urn:b"],
        "label": "A Thing",
        "embedding_text": "some text",
    }

    class _Point:
        payload = payload_for_record("my-graph", record)
        id = "pid"
        score = 0.5

    row = summarize_point(_Point())
    assert row.graph == "my-graph"
    assert row.iris == ["urn:a", "urn:b"]
    assert row.primary_uri == "urn:a"
    assert row.label == "A Thing"
    assert row.repr == "some text"


# --- iter_jsonl / chunks ---


def test_iter_jsonl_reports_line_numbers(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"g\.jsonl:2: invalid JSON"):
        list(iter_jsonl(path))


def test_iter_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert [r["a"] for r in iter_jsonl(path)] == [1, 2]


def test_chunks_batches_with_remainder():
    batches = list(chunks(iter([{"i": i} for i in range(5)]), 2))
    assert [len(b) for b in batches] == [2, 2, 1]


# --- upload_file against a fake client + stub embedder ---


class _StubEmbedder:
    def __init__(self, dim=3):
        self.dim = dim

    def embed(self, text):
        return np.ones(self.dim, dtype=np.float32)

    def embed_many(self, texts):
        return [np.ones(self.dim, dtype=np.float32) for _ in texts]


class _FakeClient:
    def __init__(self):
        self.upserted = []

    def upsert(self, collection_name, points, wait):
        self.upserted.extend(points)


class _FakeSettings:
    qdrant_collection = "C"
    qdrant_location = "http://x"


class _FakeCtx:
    def __init__(self):
        self.client = _FakeClient()
        self.embedder = _StubEmbedder()
        self.settings = _FakeSettings()


def _write_records(path, n):
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "iris": [f"urn:{i}"],
                        "label": f"L{i}",
                        "embedding_text": f"text {i}",
                    }
                )
                + "\n"
            )


def test_upload_file_upserts_all_points(tmp_path):
    path = tmp_path / "my-graph.jsonl"
    _write_records(path, 5)
    ctx = _FakeCtx()

    uploaded = upload_file(
        ctx,
        path,
        batch_size=2,
        upload_batch_size=2,
        limit=None,
        dry_run=False,
        progress_enabled=False,
        log_every=10_000,
    )

    assert uploaded == 5
    assert len(ctx.client.upserted) == 5
    # graph comes from the file stem; payload mapping applied.
    assert all(p.payload["graph"] == "my-graph" for p in ctx.client.upserted)
    assert ctx.client.upserted[0].payload["repr"] == "text 0"


def test_upload_file_dry_run_skips_upsert(tmp_path):
    path = tmp_path / "g.jsonl"
    _write_records(path, 3)
    ctx = _FakeCtx()

    uploaded = upload_file(
        ctx,
        path,
        batch_size=2,
        upload_batch_size=2,
        limit=None,
        dry_run=True,
        progress_enabled=False,
        log_every=10_000,
    )

    assert uploaded == 3
    assert ctx.client.upserted == []


def test_upload_file_honors_limit(tmp_path):
    path = tmp_path / "g.jsonl"
    _write_records(path, 10)
    ctx = _FakeCtx()

    uploaded = upload_file(
        ctx,
        path,
        batch_size=4,
        upload_batch_size=4,
        limit=3,
        dry_run=False,
        progress_enabled=False,
        log_every=10_000,
    )

    assert uploaded == 3
    assert len(ctx.client.upserted) == 3


# --- seam: single embed delegates to batch embed_many ---


def test_embed_delegates_to_embed_many():
    class _Spy(FastEmbedEmbedder):
        def __init__(self):
            self.seen = None

        def embed_many(self, texts):
            self.seen = texts
            return [np.array([1.0, 2.0], dtype=np.float32)]

    spy = _Spy()
    out = spy.embed("hello")

    assert spy.seen == ["hello"]
    assert out.tolist() == [1.0, 2.0]
