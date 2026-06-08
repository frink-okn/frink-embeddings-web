from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

from frink_embeddings_web.core.results import summarize_point
from frink_embeddings_web.indexing.models import (
    MaterializationConfiguration,
    TargetConfiguration,
)
from frink_embeddings_web.indexing.output import WorkerRecord, write_jsonl
from frink_embeddings_web.indexing.textify import (
    finish_records,
    materialize_records,
    merge_worker_record,
)
from frink_embeddings_web.indexing.upload import (
    iter_jsonl,
    payload_for_record,
)

TYPE = "http://example.org/Thing"
NAME = "http://schema.org/name"


def _config(**defaults) -> MaterializationConfiguration:
    # One target over TYPE, labelled/textified from schema:name. expansion 0
    # keeps records to the root's direct predicates so the text is predictable.
    return MaterializationConfiguration(
        targets={
            "thing": TargetConfiguration(
                type=TYPE,
                label_predicates=[NAME],
                expansion_limit=0,
            )
        },
        **defaults,
    )


def _graph_with_two_identical_things() -> Graph:
    # Two distinct IRIs whose materialized text is identical (same name, same
    # single predicate) -> they group into one record.
    g = Graph()
    for iri in ("http://example.org/a", "http://example.org/b"):
        node = URIRef(iri)
        g.add((node, RDF.type, URIRef(TYPE)))
        g.add((node, URIRef(NAME), Literal("Diabetes")))
    return g


def test_identical_text_groups_with_iri_count():
    records = materialize_records(_graph_with_two_identical_things(), _config())

    assert len(records) == 1
    record = records[0]
    assert record.iris == ["http://example.org/a", "http://example.org/b"]
    assert record.iri_count == 2


def _wr(iri: str, text: str = "same text") -> WorkerRecord:
    # digest only needs to match for records that should group together; here
    # everything shares one digest so they merge into a single OutputRecord.
    return WorkerRecord(iri=iri, label="L", embedding_text=text, digest="d")


def test_merge_keeps_smallest_n_iris_regardless_of_arrival_order():
    # Arrival order is scrambled (as it would be under parallel workers); the
    # kept set must be the lexicographically-smallest N, deterministically.
    forward: dict = {}
    for iri in ["a", "b", "c", "d", "e"]:
        merge_worker_record(forward, _wr(iri), max_iris_per_record=3)

    shuffled: dict = {}
    for iri in ["e", "c", "a", "d", "b"]:
        merge_worker_record(shuffled, _wr(iri), max_iris_per_record=3)

    (a,) = finish_records(forward)
    (b,) = finish_records(shuffled)
    assert a.iris == ["a", "b", "c"]  # smallest 3, sorted
    assert a.iris == b.iris  # order-independent
    assert a.iri_count == 5  # true total preserved
    assert b.iri_count == 5


def test_merge_dedupes_repeated_iri():
    by_digest: dict = {}
    for iri in ["a", "a", "b"]:
        merge_worker_record(by_digest, _wr(iri), max_iris_per_record=10)
    (record,) = finish_records(by_digest)
    assert record.iris == ["a", "b"]
    assert record.iri_count == 2


def test_iris_are_capped_but_count_is_total():
    records = materialize_records(
        _graph_with_two_identical_things(),
        _config(),
        max_iris_per_record=1,
    )

    assert len(records) == 1
    record = records[0]
    # Only the first (sorted) IRI is kept...
    assert record.iris == ["http://example.org/a"]
    # ...but the count reflects the true number that produced this text.
    assert record.iri_count == 2


def test_jsonl_output_round_trips_into_an_upload_payload(tmp_path):
    # The whole point of --jsonl: textify writes records that `upload` reads
    # back line by line and turns into the payload the query side expects.
    records = materialize_records(_graph_with_two_identical_things(), _config())
    out = tmp_path / "thing.jsonl"
    write_jsonl(records, out)

    read_back = list(iter_jsonl(out))
    assert len(read_back) == 1

    class _Point:
        payload = payload_for_record("thing", read_back[0])
        id = "pid"
        score = 0.9

    row = summarize_point(_Point())

    assert row.graph == "thing"
    assert row.primary_uri == "http://example.org/a"
    assert row.iris == ["http://example.org/a", "http://example.org/b"]
    assert row.label == "Diabetes"
    assert row.repr == records[0].embedding_text
    assert row.iri_count == 2
