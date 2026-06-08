from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

from frink_embeddings_web.core.results import summarize_point
from frink_embeddings_web.indexing.index import (
    materialize_records,
    write_jsonl,
)
from frink_embeddings_web.indexing.models import (
    MaterializationConfiguration,
    TargetConfiguration,
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
