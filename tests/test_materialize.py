from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from frink_embeddings_web.indexing.index import (
    build_embedding_text,
    display_label,
    effective_label_predicates,
    fallback_label,
    humanize,
    materialize_records,
    stable_score,
    walk_graph,
)
from frink_embeddings_web.indexing.models import (
    GraphConfiguration,
    MaterializationConfiguration,
)
from frink_embeddings_web.indexing.sample import sample_targets, sample_types

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Graph:
    return Graph().parse(FIXTURES / name, format="turtle")


def test_humanize_text():
    assert humanize("hello_world") == "hello world"
    assert humanize("strip-till") == "strip till"
    assert humanize("ProjectScenario") == "Project Scenario"


def test_url_fallback():
    assert fallback_label("http://example.com/FirstName") == "First Name"
    assert fallback_label(
        "http://example.com/ontology#hasAttribute"
    ) == "has Attribute"


def test_config_merges_defaults_and_target_overrides():
    config = MaterializationConfiguration.model_validate(
        {
            "defaults": {
                "label_predicates": ["http://example.com/defaultLabel"],
                "label_fields": {
                    "default_id": "http://example.com/defaultId"
                },
                "ignore_predicates": ["http://example.com/ignoreDefault"],
                "predicate_limit": 3,
                "expansion_limit": 1,
                "include_rdfs_label": False,
            },
            "targets": {
                "thing": {
                    "type": "http://example.com/Thing",
                    "label_predicates": ["http://example.com/targetLabel"],
                    "label_template": "{name}: {default_id}",
                    "label_fields": {
                        "name": "http://example.com/name",
                    },
                    "ignore_predicates": ["http://example.com/ignoreTarget"],
                    "predicate_limit": 2,
                    "expansion_limit": 2,
                    "include_rdfs_label": True,
                }
            },
        }
    )

    target = config.for_target("thing")

    assert target.label_predicates == [
        "http://example.com/defaultLabel",
        "http://example.com/targetLabel",
    ]
    assert target.label_template == "{name}: {default_id}"
    assert target.label_fields == {
        "default_id": "http://example.com/defaultId",
        "name": "http://example.com/name",
    }
    assert target.ignore_predicates == [
        "http://example.com/ignoreDefault",
        "http://example.com/ignoreTarget",
    ]
    assert target.predicate_limit == 2
    assert target.expansion_limit == 2
    assert target.include_rdfs_label is True


def test_effective_label_predicates_can_include_rdfs_label():
    config = GraphConfiguration(
        label_predicates=["http://example.com/name"],
        include_rdfs_label=True,
    )

    assert effective_label_predicates(config) == [
        "http://example.com/name",
        str(RDFS.label),
    ]


def test_effective_label_predicates_can_exclude_rdfs_label():
    config = GraphConfiguration(
        label_predicates=["http://example.com/name"],
        include_rdfs_label=False,
    )

    assert effective_label_predicates(config) == ["http://example.com/name"]


def test_walk_graph_uses_ignore_predicates_limit_and_depth():
    graph = load_fixture("walk_graph.ttl")
    root = URIRef("http://example.com/root")
    pred = URIRef("http://example.com/hasPart")
    ignored = URIRef("http://example.com/ignored")
    leaf_pred = URIRef("http://example.com/leaf")
    objects = [
        URIRef("http://example.com/objectA"),
        URIRef("http://example.com/objectB"),
        URIRef("http://example.com/objectC"),
    ]

    config = GraphConfiguration(
        ignore_predicates=[str(ignored)],
        predicate_limit=2,
        expansion_limit=1,
    )

    triples = list(walk_graph(graph, root, config))
    selected_objects = sorted(
        objects,
        key=lambda o: stable_score(root, pred, o),
    )

    assert (0, root, ignored, Literal("skip me")) not in triples
    assert [triple[3] for triple in triples if triple[1] == root] == (
        selected_objects[:2]
    )
    assert any(level == 1 and p == leaf_pred for level, _, p, _ in triples)


def test_build_embedding_text_formats_labels_literals_and_nested_nodes():
    graph = load_fixture("embedding_text.ttl")
    root = URIRef("http://example.com/root")
    config = GraphConfiguration(expansion_limit=1)

    text = build_embedding_text(graph, root, config)

    assert "entity: Root label" in text
    assert "related predicate: Related label" in text
    assert "has score: 42" in text
    assert "  nested name: Nested literal" in text


def test_display_label_uses_template_fields_with_fallback():
    graph = load_fixture("embedding_text.ttl")
    root = URIRef("http://example.com/root")
    config = GraphConfiguration(
        label_template="{name}: {score}",
        label_fields={
            "name": str(RDFS.label),
            "score": "http://example.com/has-score",
        },
    )

    assert display_label(graph, root, config) == "Root label: 42"


def test_materialize_records_groups_duplicate_text_by_iris():
    graph = load_fixture("dedupe.ttl")
    root_type = URIRef("http://example.com/Thing")
    root_a = URIRef("http://example.com/a")
    root_b = URIRef("http://example.com/b")
    root_c = URIRef("http://example.com/c")

    config = MaterializationConfiguration.model_validate(
        {
            "targets": {
                "thing": {
                    "type": str(root_type),
                    "ignore_predicates": [str(RDF.type)],
                    "include_rdfs_label": False,
                }
            }
        }
    )

    records = materialize_records(graph, config)

    assert len(records) == 2
    grouped = {record.embedding_text: record.iris for record in records}
    assert grouped["value: same"] == [str(root_a), str(root_b)]
    assert grouped["value: different"] == [str(root_c)]
    labels = {record.embedding_text: record.label for record in records}
    assert labels["value: same"] == "a"


def test_sample_types_reports_literal_and_object_predicate_evidence():
    graph = load_fixture("sample_types.ttl")

    records = sample_types(graph, limit=2, values_limit=1)

    by_type = {record.type: record for record in records}
    treatment = by_type["http://example.com/Treatment"]

    assert treatment.label == "Treatment"
    assert treatment.count == 2
    assert treatment.sample_iris == [
        "http://example.com/treatmentA",
        "http://example.com/treatmentB",
    ]

    predicates = {
        predicate.predicate: predicate
        for predicate in treatment.literal_predicates
    }
    assert predicates["http://schema.org/name"].label == "name"
    assert predicates["http://schema.org/name"].count == 2
    assert predicates["http://schema.org/name"].values == ["Treatment A"]
    assert predicates["http://purl.org/dc/terms/description"].label == (
        "description"
    )
    assert "http://example.com/linksTo" not in predicates

    object_predicates = {
        predicate.predicate: predicate
        for predicate in treatment.object_predicates
    }
    links_to = object_predicates["http://example.com/linksTo"]

    assert links_to.label == "links to"
    assert links_to.count == 2
    assert links_to.object_types[0].type == "http://example.com/Project"
    assert links_to.object_types[0].label == "Project"
    assert links_to.object_types[0].count == 2

    label_predicates = {
        predicate.predicate: predicate
        for predicate in links_to.object_label_predicates
    }
    assert label_predicates["http://schema.org/name"].values == ["Project A"]
    assert label_predicates["http://purl.org/dc/terms/description"].values == [
        "A project"
    ]
    assert links_to.sample_objects[0].iri == "http://example.com/projectA"
    assert links_to.sample_objects[0].label == "project A"
    assert links_to.sample_objects[0].types == ["http://example.com/Project"]


def test_sample_targets_uses_configured_materialization():
    graph = load_fixture("dedupe.ttl")
    root_type = URIRef("http://example.com/Thing")

    config = MaterializationConfiguration.model_validate(
        {
            "targets": {
                "thing": {
                    "type": str(root_type),
                    "ignore_predicates": [str(RDF.type)],
                    "include_rdfs_label": False,
                }
            }
        }
    )

    records = sample_targets(graph, config, limit=1)

    assert len(records) == 1
    assert records[0].target == "thing"
    assert records[0].type == str(root_type)
    assert len(records[0].records) == 1
    assert records[0].records[0].embedding_text == "value: same"
