import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

from .index import (
    OutputRecord,
    best_label,
    materialize_records,
    predicate_text,
)
from .models import GraphConfiguration, MaterializationConfiguration


@dataclass
class LiteralPredicateSample:
    predicate: str
    label: str
    count: int
    values: list[str]


@dataclass
class TypeSampleRecord:
    type: str
    label: str
    count: int
    sample_iris: list[str]
    literal_predicates: list[LiteralPredicateSample]


@dataclass
class TargetSampleRecord:
    target: str
    type: str
    records: list[OutputRecord]


def sample_types(
    graph: Graph,
    limit: int = 5,
    values_limit: int = 3,
) -> list[TypeSampleRecord]:
    counts: defaultdict[str, int] = defaultdict(int)
    samples: defaultdict[str, list[str]] = defaultdict(list)

    for subject, type_node in graph.subject_objects(RDF.type):
        if not isinstance(subject, URIRef) or not isinstance(type_node, URIRef):
            continue

        type_iri = str(type_node)
        counts[type_iri] += 1
        if len(samples[type_iri]) < limit:
            samples[type_iri].append(str(subject))

    config = GraphConfiguration()
    records: list[TypeSampleRecord] = []

    for type_iri in sorted(counts):
        predicate_counts: defaultdict[str, int] = defaultdict(int)
        predicate_values: defaultdict[str, list[str]] = defaultdict(list)

        for sample_iri in samples[type_iri]:
            for pred, obj in graph.predicate_objects(URIRef(sample_iri)):
                if pred == RDF.type or not isinstance(pred, URIRef):
                    continue
                if not isinstance(obj, Literal):
                    continue

                pred_iri = str(pred)
                predicate_counts[pred_iri] += 1
                values = predicate_values[pred_iri]
                obj_text = str(obj)
                if obj_text not in values and len(values) < values_limit:
                    values.append(obj_text)

        literal_predicates = [
            LiteralPredicateSample(
                predicate=pred_iri,
                label=predicate_text(graph, URIRef(pred_iri), config),
                count=count,
                values=predicate_values[pred_iri],
            )
            for pred_iri, count in sorted(
                predicate_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

        records.append(
            TypeSampleRecord(
                type=type_iri,
                label=best_label(
                    graph,
                    URIRef(type_iri),
                    config,
                    use_fallback=True,
                ),
                count=counts[type_iri],
                sample_iris=samples[type_iri],
                literal_predicates=literal_predicates,
            )
        )

    return records


def sample_targets(
    graph: Graph,
    config: MaterializationConfiguration,
    limit: int = 5,
) -> list[TargetSampleRecord]:
    records = []

    for target in config.targets:
        target_config = config.for_target(target)
        records.append(
            TargetSampleRecord(
                target=target,
                type=target_config.type,
                records=materialize_records(
                    graph,
                    config,
                    target=target,
                    limit=limit,
                ),
            )
        )

    return records


def write_sample_types_json(
    records: Iterable[TypeSampleRecord],
    output_path: Path,
):
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in records],
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_sample_types_text(
    records: Iterable[TypeSampleRecord],
    output_path: Path,
):
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(f"type: {record.type}\n")
            f.write(f"label: {record.label}\n")
            f.write(f"count: {record.count}\n")
            f.write("sample iris:\n")
            for iri in record.sample_iris:
                f.write(f"- {iri}\n")
            f.write("literal predicates:\n")
            for pred in record.literal_predicates:
                f.write(
                    f"- {pred.label} ({pred.predicate}) "
                    f"[count={pred.count}]\n"
                )
                for value in pred.values:
                    f.write(f"  - {value}\n")
            f.write("\n---\n\n")


def write_sample_targets_json(
    records: Iterable[TargetSampleRecord],
    output_path: Path,
):
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in records],
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_sample_targets_text(
    records: Iterable[TargetSampleRecord],
    output_path: Path,
):
    with output_path.open("w", encoding="utf-8") as f:
        for target_record in records:
            f.write(f"target: {target_record.target}\n")
            f.write(f"type: {target_record.type}\n\n")
            for record in target_record.records:
                f.write("iris:\n")
                for iri in record.iris:
                    f.write(f"- {iri}\n")
                f.write("\n")
                f.write(record.embedding_text)
                f.write("\n\n")
            f.write("---\n\n")
