import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TypeVar

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

from .models import GraphConfiguration, MaterializationConfiguration
from .output import OutputRecord
from .reader import graph_reader
from .textify import (
    Textifier,
    finish_records,
    merge_worker_record,
)

T = TypeVar("T")

# Seed used for graphs whose identity cannot be read (non-HDT stores, mainly
# in tests). Any constant will do; it only has to be stable.
FALLBACK_SEED = 0


def graph_seed(graph: Graph) -> int:
    """A reproducible sampling seed derived from the graph's own identity.

    Read from the HDT header, which costs nothing, rather than hashing the
    file: the samples then stay the same when the file is renamed, copied, or
    moved, and change when the graph itself does.
    """
    document = getattr(graph.store, "hdt_document", None)
    if document is None:
        return FALLBACK_SEED

    identity = "|".join(
        str(value)
        for value in (
            document.total_triples,
            document.nb_subjects,
            document.nb_predicates,
            document.nb_objects,
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def reservoir_sample(
    items: Iterable[T], k: int, rng: random.Random
) -> list[T]:
    """A uniform `k`-sample of `items`, whose length need not be known.

    Vitter's Algorithm R: keep the first k, then admit the i-th item with
    probability k/(i+1), evicting a uniformly chosen incumbent. Every item
    ends up in the reservoir with probability k/n for the final n, while
    memory stays O(k) -- which is the point on a graph with millions of
    subjects per type.
    """
    reservoir: list[T] = []
    for i, item in enumerate(items):
        if i < k:
            reservoir.append(item)
        elif (j := rng.randrange(i + 1)) < k:
            reservoir[j] = item
    return reservoir


@dataclass
class LiteralPredicateSample:
    predicate: str
    label: str
    count: int
    values: list[str]


@dataclass
class ObjectTypeSample:
    type: str
    label: str
    count: int


@dataclass
class ObjectSample:
    iri: str
    label: str | None
    types: list[str]


@dataclass
class ObjectPredicateSample:
    predicate: str
    label: str
    count: int
    object_types: list[ObjectTypeSample]
    object_label_predicates: list[LiteralPredicateSample]
    sample_objects: list[ObjectSample]


@dataclass
class TypeSampleRecord:
    type: str
    label: str
    count: int
    sample_iris: list[str]
    literal_predicates: list[LiteralPredicateSample]
    object_predicates: list[ObjectPredicateSample]


@dataclass
class TargetSampleRecord:
    target: str
    type: str
    records: list[OutputRecord]


def sample_types(
    graph: Graph,
    limit: int = 5,
    values_limit: int = 3,
    seed: int | None = None,
) -> list[TypeSampleRecord]:
    """Survey the graph's types, with a uniform sample of each type's subjects.

    Sampling is seeded (from the graph's identity unless `seed` is given), so
    a run over the same graph always reports the same exemplars.
    """
    # Reproducibility, not secrecy: a seeded Mersenne Twister is the point.
    rng = random.Random(  # noqa: S311
        graph_seed(graph) if seed is None else seed
    )

    counts: defaultdict[str, int] = defaultdict(int)
    samples: defaultdict[str, list[str]] = defaultdict(list)

    # One reservoir per type, sharing this single pass. Each type's counter is
    # its own `i`, so types spanning five orders of magnitude -- 3.9M classes
    # and 63 ontologies -- self-adjust without knowing their size in advance.
    for subject, type_node in graph.subject_objects(RDF.type):
        if not isinstance(subject, URIRef) or not isinstance(type_node, URIRef):
            continue

        type_iri = str(type_node)
        seen = counts[type_iri]
        counts[type_iri] = seen + 1

        bucket = samples[type_iri]
        if seen < limit:
            bucket.append(str(subject))
        elif (j := rng.randrange(seen + 1)) < limit:
            bucket[j] = str(subject)

    # Present each reservoir in a stable order; the *set* is what the sampling
    # makes uniform.
    for bucket in samples.values():
        bucket.sort()

    config = GraphConfiguration()
    textifier = Textifier(graph_reader(graph))
    records: list[TypeSampleRecord] = []

    for type_iri in sorted(counts):
        predicate_counts: defaultdict[str, int] = defaultdict(int)
        predicate_values: defaultdict[str, list[str]] = defaultdict(list)
        object_counts: defaultdict[str, int] = defaultdict(int)
        object_type_counts: defaultdict[
            str,
            defaultdict[str, int],
        ] = defaultdict(lambda: defaultdict(int))
        object_label_counts: defaultdict[
            str,
            defaultdict[str, int],
        ] = defaultdict(lambda: defaultdict(int))
        object_label_values: defaultdict[
            str,
            defaultdict[str, list[str]],
        ] = defaultdict(lambda: defaultdict(list))
        object_samples: defaultdict[str, list[ObjectSample]] = defaultdict(list)

        for sample_iri in samples[type_iri]:
            for pred, obj in graph.predicate_objects(URIRef(sample_iri)):
                if pred == RDF.type or not isinstance(pred, URIRef):
                    continue
                pred_iri = str(pred)

                if isinstance(obj, Literal):
                    predicate_counts[pred_iri] += 1
                    values = predicate_values[pred_iri]
                    obj_text = str(obj)
                    if obj_text not in values and len(values) < values_limit:
                        values.append(obj_text)
                    continue

                if not isinstance(obj, URIRef):
                    continue

                object_counts[pred_iri] += 1
                object_type_iris = [
                    str(object_type)
                    for object_type in graph.objects(obj, RDF.type)
                    if isinstance(object_type, URIRef)
                ]
                for object_type_iri in object_type_iris:
                    object_type_counts[pred_iri][object_type_iri] += 1

                for label_pred, label_value in graph.predicate_objects(obj):
                    if (
                        label_pred == RDF.type
                        or not isinstance(label_pred, URIRef)
                        or not isinstance(label_value, Literal)
                    ):
                        continue
                    label_pred_iri = str(label_pred)
                    object_label_counts[pred_iri][label_pred_iri] += 1
                    values = object_label_values[pred_iri][label_pred_iri]
                    label_text = str(label_value)
                    if label_text not in values and len(values) < values_limit:
                        values.append(label_text)

                samples_for_pred = object_samples[pred_iri]
                seen_sample_object = any(
                    sample.iri == str(obj) for sample in samples_for_pred
                )
                if not seen_sample_object and len(samples_for_pred) < limit:
                    samples_for_pred.append(
                        ObjectSample(
                            iri=str(obj),
                            label=textifier.best_label(obj, config),
                            types=sorted(object_type_iris),
                        )
                    )

        literal_predicates = [
            LiteralPredicateSample(
                predicate=pred_iri,
                label=textifier.predicate_text(URIRef(pred_iri), config),
                count=count,
                values=predicate_values[pred_iri],
            )
            for pred_iri, count in sorted(
                predicate_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

        object_predicates = [
            ObjectPredicateSample(
                predicate=pred_iri,
                label=textifier.predicate_text(URIRef(pred_iri), config),
                count=count,
                object_types=[
                    ObjectTypeSample(
                        type=object_type_iri,
                        label=textifier.best_label(
                            URIRef(object_type_iri),
                            config,
                            use_fallback=True,
                        ),
                        count=object_type_count,
                    )
                    for object_type_iri, object_type_count in sorted(
                        object_type_counts[pred_iri].items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
                object_label_predicates=[
                    LiteralPredicateSample(
                        predicate=label_pred_iri,
                        label=textifier.predicate_text(
                            URIRef(label_pred_iri),
                            config,
                        ),
                        count=label_pred_count,
                        values=object_label_values[pred_iri][label_pred_iri],
                    )
                    for label_pred_iri, label_pred_count in sorted(
                        object_label_counts[pred_iri].items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
                sample_objects=object_samples[pred_iri],
            )
            for pred_iri, count in sorted(
                object_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

        records.append(
            TypeSampleRecord(
                type=type_iri,
                label=textifier.best_label(
                    URIRef(type_iri),
                    config,
                    use_fallback=True,
                ),
                count=counts[type_iri],
                sample_iris=samples[type_iri],
                literal_predicates=literal_predicates,
                object_predicates=object_predicates,
            )
        )

    return records


def sample_targets(
    graph: Graph,
    config: MaterializationConfiguration,
    limit: int = 5,
    seed: int | None = None,
) -> list[TargetSampleRecord]:
    """Materialize a uniform sample of each target's roots.

    Unlike a full `textify` run this samples rather than taking the first
    `limit` roots, so the sampled documents represent the whole target rather
    than whichever IRIs sort first. Seeded like `sample_types`.
    """
    # Reproducibility, not secrecy: a seeded Mersenne Twister is the point.
    rng = random.Random(  # noqa: S311
        graph_seed(graph) if seed is None else seed
    )
    reader = graph_reader(graph)
    textifier = Textifier(reader, config)
    records = []

    for target in config.targets:
        target_config = config.for_target(target)
        roots = reservoir_sample(
            reader.root_iris(target_config.type), limit, rng
        )

        by_digest: dict[str, OutputRecord] = {}
        for iri in sorted(roots):
            merge_worker_record(
                by_digest, textifier.materialize_one(iri, target_config)
            )

        records.append(
            TargetSampleRecord(
                target=target,
                type=target_config.type,
                records=finish_records(by_digest),
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
                    f"- {pred.label} ({pred.predicate}) [count={pred.count}]\n"
                )
                for value in pred.values:
                    f.write(f"  - {value}\n")
            f.write("object predicates:\n")
            for pred in record.object_predicates:
                f.write(
                    f"- {pred.label} ({pred.predicate}) [count={pred.count}]\n"
                )
                f.write("  object types:\n")
                for object_type in pred.object_types:
                    f.write(
                        f"  - {object_type.label} ({object_type.type}) "
                        f"[count={object_type.count}]\n"
                    )
                f.write("  object label predicates:\n")
                for label_pred in pred.object_label_predicates:
                    f.write(
                        f"  - {label_pred.label} "
                        f"({label_pred.predicate}) "
                        f"[count={label_pred.count}]\n"
                    )
                    for value in label_pred.values:
                        f.write(f"    - {value}\n")
                f.write("  sample objects:\n")
                for obj in pred.sample_objects:
                    f.write(f"  - {obj.iri}")
                    if obj.label:
                        f.write(f" [{obj.label}]")
                    f.write("\n")
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
                f.write(f"label: {record.label}\n")
                f.write("iris:\n")
                for iri in record.iris:
                    f.write(f"- {iri}\n")
                f.write("\n")
                f.write(record.embedding_text)
                f.write("\n\n")
            f.write("---\n\n")
