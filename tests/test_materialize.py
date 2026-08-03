import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from okn_embeddings.indexing.models import (
    GraphConfiguration,
    MaterializationConfiguration,
)
from okn_embeddings.indexing.reader import graph_reader
from okn_embeddings.indexing.sample import (
    FALLBACK_SEED,
    graph_seed,
    reservoir_sample,
    sample_targets,
    sample_types,
)
from okn_embeddings.indexing.text import (
    effective_label_predicates,
    fallback_label,
    humanize,
    stable_score,
)
from okn_embeddings.indexing.textify import (
    Textifier,
    materialize_records,
)

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
                "ignore_predicates": ["http://example.com/ignoreDefault"],
                "predicate_limit": 3,
                "expansion_limit": 1,
                "include_rdfs_label": False,
            },
            "targets": {
                "thing": {
                    "type": "http://example.com/Thing",
                    "label_profile": "thing_label",
                    "label_predicates": ["http://example.com/targetLabel"],
                    "label_template": "{name}",
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
    assert target.label_profile == "thing_label"
    assert target.label_template == "{name}"
    assert target.label_fields == {
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

    triples = list(Textifier(graph_reader(graph)).walk(root, config))
    selected_objects = sorted(
        objects,
        key=lambda o: stable_score(root, pred, o),
    )

    # The ignored predicate never appears.
    assert all(p.value != str(ignored) for _, _, p, _ in triples)
    # predicate_limit keeps the two lowest-scoring objects at the root.
    root_objects = [o.value for level, _, _, o in triples if level == 0]
    assert root_objects == [str(o) for o in selected_objects[:2]]
    # Expansion reaches the leaf predicate one hop down.
    assert any(
        level == 1 and p.value == str(leaf_pred) for level, _, p, _ in triples
    )


def test_walk_graph_include_predicates_is_an_allowlist():
    graph = load_fixture("walk_graph.ttl")
    root = URIRef("http://example.com/root")
    pred = URIRef("http://example.com/hasPart")
    ignored = URIRef("http://example.com/ignored")

    config = GraphConfiguration(
        include_predicates=[str(ignored)],
        expansion_limit=0,
    )

    triples = list(Textifier(graph_reader(graph)).walk(root, config))

    # Only the allowed predicate survives, even though it is the one the
    # blacklist test skips.
    assert {p.value for _, _, p, _ in triples} == {str(ignored)}
    assert all(p.value != str(pred) for _, _, p, _ in triples)


def test_walk_graph_ignore_predicates_still_applies_within_the_allowlist():
    graph = load_fixture("walk_graph.ttl")
    root = URIRef("http://example.com/root")
    pred = URIRef("http://example.com/hasPart")
    ignored = URIRef("http://example.com/ignored")

    config = GraphConfiguration(
        include_predicates=[str(pred), str(ignored)],
        ignore_predicates=[str(ignored)],
        expansion_limit=0,
    )

    triples = list(Textifier(graph_reader(graph)).walk(root, config))

    assert {p.value for _, _, p, _ in triples} == {str(pred)}


def test_walk_graph_empty_include_predicates_allows_everything():
    graph = load_fixture("walk_graph.ttl")
    root = URIRef("http://example.com/root")

    config = GraphConfiguration(expansion_limit=0)

    triples = list(Textifier(graph_reader(graph)).walk(root, config))

    assert len({p.value for _, _, p, _ in triples}) == 2


def test_config_merges_include_predicates_from_defaults_and_target():
    config = MaterializationConfiguration.model_validate(
        {
            "defaults": {
                "include_predicates": ["http://example.com/keepDefault"],
            },
            "targets": {
                "thing": {
                    "type": "http://example.com/Thing",
                    "include_predicates": ["http://example.com/keepTarget"],
                },
            },
        }
    )

    assert config.for_target("thing").include_predicates == [
        "http://example.com/keepDefault",
        "http://example.com/keepTarget",
    ]


def test_build_embedding_text_formats_labels_literals_and_nested_nodes():
    graph = load_fixture("embedding_text.ttl")
    root = URIRef("http://example.com/root")
    config = GraphConfiguration(expansion_limit=1)

    text = Textifier(graph_reader(graph)).build_embedding_text(root, config)

    assert "label: Root label" in text
    assert "related predicate: Related label" in text
    assert "has score: 42" in text
    assert "nested name: Nested literal" not in text


def test_display_label_uses_target_template_fields_with_fallback():
    graph = load_fixture("embedding_text.ttl")
    root = URIRef("http://example.com/root")
    config = MaterializationConfiguration.model_validate(
        {
            "targets": {
                "thing": {
                    "type": "http://example.com/Thing",
                    "label_template": "{name}: {score}",
                    "label_fields": {
                        "name": str(RDFS.label),
                        "score": "http://example.com/has-score",
                    },
                }
            }
        }
    )
    target = config.for_target("thing")

    assert (
        Textifier(graph_reader(graph), config).display_label(root, target)
        == "Root label: 42"
    )


def test_display_label_uses_label_profile_for_non_target_nodes():
    graph = load_fixture("embedding_text.ttl")
    related = URIRef("http://example.com/relatedThing")
    config = MaterializationConfiguration.model_validate(
        {
            "label_profiles": {
                "related": {
                    "type": "http://example.com/Related",
                    "template": "profile: {name}",
                    "fields": {
                        "name": str(RDFS.label),
                    },
                }
            },
            "targets": {
                "thing": {
                    "type": "http://example.com/Thing",
                }
            },
        }
    )
    graph.add((related, RDF.type, URIRef("http://example.com/Related")))

    assert (
        Textifier(graph_reader(graph), config).display_label(
            related, config.for_target("thing")
        )
        == "profile: Related label"
    )


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

    assert len(records) == 3
    grouped = {record.embedding_text: record.iris for record in records}
    assert grouped["label: a\nvalue: same"] == [str(root_a)]
    assert grouped["label: b\nvalue: same"] == [str(root_b)]
    assert grouped["label: c\nvalue: different"] == [str(root_c)]
    labels = {record.embedding_text: record.label for record in records}
    assert labels["label: a\nvalue: same"] == "a"


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

    records = sample_targets(graph, config, limit=1, seed=0)

    assert len(records) == 1
    assert records[0].target == "thing"
    assert records[0].type == str(root_type)
    assert len(records[0].records) == 1
    # Which root is sampled is up to the RNG; whichever it is, the record is
    # materialized through the configured target.
    assert records[0].records[0].embedding_text in {
        "label: a\nvalue: same",
        "label: b\nvalue: same",
        "label: c\nvalue: different",
    }


# --- sampling: uniform, seeded, reproducible -----------------------------


def _many_things(n: int) -> Graph:
    """A graph of `n` ex:Thing roots named in iteration order t00, t01, …"""
    graph = Graph()
    thing = URIRef("http://example.com/Thing")
    for i in range(n):
        subject = URIRef(f"http://example.com/t{i:02d}")
        graph.add((subject, RDF.type, thing))
        graph.add((subject, RDFS.label, Literal(f"thing {i:02d}")))
    return graph


def test_reservoir_sample_keeps_k_items_from_the_stream():
    picked = reservoir_sample(range(1000), 5, random.Random(1))

    assert len(picked) == 5
    assert len(set(picked)) == 5
    assert all(0 <= item < 1000 for item in picked)


def test_reservoir_sample_returns_everything_when_shorter_than_k():
    assert sorted(reservoir_sample(range(3), 10, random.Random(1))) == [
        0,
        1,
        2,
    ]


def test_reservoir_sample_is_uniform():
    # Every item should land in the reservoir about k/n of the time. With
    # n=10, k=2 and 4000 draws, each item's expected share is 20%.
    rng = random.Random(7)
    hits = Counter()
    trials = 4000
    for _ in range(trials):
        hits.update(reservoir_sample(range(10), 2, rng))

    shares = [hits[i] / trials for i in range(10)]
    assert all(0.15 < share < 0.25 for share in shares), shares


def test_sample_types_looks_past_the_first_n_subjects():
    graph = _many_things(60)

    records = sample_types(graph, limit=3, seed=99)
    sampled = set(records[0].sample_iris)

    assert len(sampled) == 3
    # The old behavior kept exactly t00, t01, t02 -- the lexicographically
    # first roots. A uniform sample of 3 from 60 essentially never does.
    assert sampled != {
        "http://example.com/t00",
        "http://example.com/t01",
        "http://example.com/t02",
    }


def test_sample_types_is_reproducible_for_a_seed():
    graph = _many_things(60)

    first = sample_types(graph, limit=3, seed=99)[0].sample_iris
    again = sample_types(graph, limit=3, seed=99)[0].sample_iris
    other = sample_types(graph, limit=3, seed=1234)[0].sample_iris

    assert first == again
    assert first != other


def test_sample_targets_looks_past_the_first_n_roots():
    graph = _many_things(60)
    config = MaterializationConfiguration.model_validate(
        {
            "targets": {
                "thing": {
                    "type": "http://example.com/Thing",
                    "ignore_predicates": [str(RDF.type)],
                }
            }
        }
    )

    records = sample_targets(graph, config, limit=3, seed=99)[0].records
    labels = {record.label for record in records}

    assert len(labels) == 3
    assert labels != {"thing 00", "thing 01", "thing 02"}


def test_graph_seed_falls_back_without_an_hdt_document():
    assert graph_seed(Graph()) == FALLBACK_SEED


def test_graph_seed_is_derived_from_hdt_header_stats():
    def fake(total_triples):
        return SimpleNamespace(
            store=SimpleNamespace(
                hdt_document=SimpleNamespace(
                    total_triples=total_triples,
                    nb_subjects=17,
                    nb_predicates=3,
                    nb_objects=25,
                )
            )
        )

    # Same graph identity -> same seed; a different graph -> different seed.
    assert graph_seed(fake(600)) == graph_seed(fake(600))
    assert graph_seed(fake(600)) != graph_seed(fake(601))
    assert graph_seed(fake(600)) != FALLBACK_SEED
