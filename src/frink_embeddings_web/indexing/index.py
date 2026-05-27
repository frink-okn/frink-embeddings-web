import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generator, Iterable

import typer
from rdflib import BNode, Graph, Literal, Node, URIRef
from rdflib.namespace import RDF
from rdflib_hdt import HDTStore

from .models import (
    GraphConfiguration,
    LabelProfileConfiguration,
    MaterializationConfiguration,
)

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

app = typer.Typer()


@dataclass
class OutputRecord:
    iris: list[str]
    label: str
    embedding_text: str


def load_graph(hdt_file: Path):
    store = HDTStore(str(hdt_file))
    return Graph(store=store)


def humanize(text: str) -> str:
    text = text.replace("_", " ").replace("-", " ")
    out = []
    prev_lower = False

    for c in text:
        if prev_lower and c.isupper():
            out.append(" ")
        out.append(c)
        prev_lower = c.islower()

    return " ".join("".join(out).split())


def iri_fragment(iri: str) -> str:
    if "#" in iri:
        return iri.rsplit("#", 1)[1]
    if "/" in iri:
        return iri.rsplit("/", 1)[1]
    return iri


def fallback_label(iri: str) -> str:
    return humanize(iri_fragment(iri))


def effective_label_predicates(config: GraphConfiguration) -> list[str]:
    predicates = [*config.label_predicates]
    if config.include_rdfs_label and RDFS_LABEL not in predicates:
        predicates.append(RDFS_LABEL)
    return predicates


def first_literal(graph: Graph, s: Node, ps: Iterable[str]):
    for p in ps:
        for o in graph.objects(s, URIRef(p)):
            if isinstance(o, Literal):
                return str(o)
    return None


def best_label(
    graph: Graph,
    node: Any,
    config: GraphConfiguration | None = None,
    use_fallback=True,
    use_humanize=True,
):
    if isinstance(node, Literal):
        return str(node)

    label_predicates = (
        effective_label_predicates(config)
        if config is not None
        else (RDFS_LABEL,)
    )

    if isinstance(node, Node):
        label = first_literal(graph, node, label_predicates)
        if label:
            return label

    if isinstance(node, URIRef) and use_fallback:
        return humanize(iri_fragment(str(node))) if use_humanize else str(node)

    return None


def predicate_text(graph: Graph, pred: URIRef, config: GraphConfiguration):
    label = best_label(graph, pred, config)
    if not label:
        label = fallback_label(str(pred))
    return humanize(label).lower()


def normalize_label(text: str) -> str:
    return " ".join(text.split())


def stable_score(root: Node, pred: Node, obj: Node) -> str:
    text = f"{root}\t{pred}\t{obj}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def walk_graph(
    graph: Graph,
    root: Node,
    config: GraphConfiguration,
    expansion_level=0,
) -> Generator[tuple[int, Node, Node, Node], None, None]:
    if config.ignore_predicates is None:
        ignore_predicates = set()
    else:
        ignore_predicates = set(config.ignore_predicates)

    objects_by_predicate: defaultdict[Node, list[Node]] = defaultdict(list)

    for p, o in graph.predicate_objects(root):
        p_str = str(p)
        if p_str in ignore_predicates:
            continue
        objects_by_predicate[p].append(o)

    for p in sorted(objects_by_predicate, key=str):
        objects = objects_by_predicate[p]
        if config.predicate_limit is not None:
            objects = sorted(
                objects,
                key=lambda o: stable_score(root, p, o),
            )[: config.predicate_limit]

        for o in objects:
            yield expansion_level, root, p, o

            if (
                not isinstance(o, Literal)
                and expansion_level < config.expansion_limit
            ):
                yield from walk_graph(
                    graph,
                    o,
                    config,
                    expansion_level=expansion_level + 1,
                )


def build_graph(
    graph: Graph,
    root: Node,
    config: GraphConfiguration,
) -> Graph:
    g = Graph()

    for _, s, p, o in walk_graph(graph, root, config):
        g.add((s, p, o))

    return g


def build_embedding_text(
    graph: Graph,
    root: Node,
    config: GraphConfiguration,
    label: str | None = None,
    materialization_config: MaterializationConfiguration | None = None,
) -> str:
    lines: list[str] = []

    if label is None:
        label = display_label(
            graph,
            root,
            config,
            materialization_config=materialization_config,
        )
    if label:
        lines.append(f"label: {label}")

    for level, _, p, o in walk_graph(graph, root, config):
        if level > 0:
            continue
        if not isinstance(p, URIRef):
            continue

        pred_txt = predicate_text(graph, p, config)
        obj_txt = display_label(
            graph,
            o,
            config,
            materialization_config=materialization_config,
            use_target_template=False,
        )

        if not obj_txt or isinstance(o, BNode):
            continue

        lines.append(f"{pred_txt}: {obj_txt}")

    return "\n".join(lines)


def first_direct_value(
    graph: Graph,
    root: Node,
    predicate_iri: str,
    config: GraphConfiguration,
    materialization_config: MaterializationConfiguration | None = None,
) -> str | None:
    values = []

    for obj in graph.objects(root, URIRef(predicate_iri)):
        if isinstance(obj, BNode):
            continue
        label = display_label(
            graph,
            obj,
            config,
            materialization_config=materialization_config,
            use_target_template=False,
        )
        if label:
            values.append(label)

    if not values:
        return None

    return sorted(values)[0]


def render_template(
    graph: Graph,
    root: Node,
    template: str,
    fields: dict[str, str],
    config: GraphConfiguration,
    materialization_config: MaterializationConfiguration | None = None,
) -> str | None:
    def replace(match: re.Match[str]) -> str:
        field = match.group(1).strip()
        predicate_iri = fields.get(field)
        if predicate_iri is None:
            return ""
        return (
            first_direct_value(
                graph,
                root,
                predicate_iri,
                config,
                materialization_config=materialization_config,
            )
            or ""
        )

    label = re.sub(r"\{([^{}]+)\}", replace, template)
    label = normalize_label(label)
    return label or None


def render_label_template(
    graph: Graph,
    root: Node,
    config: GraphConfiguration,
    materialization_config: MaterializationConfiguration | None = None,
) -> str | None:
    template = getattr(config, "label_template", None)
    fields = getattr(config, "label_fields", {})
    if not template:
        return None
    return render_template(
        graph,
        root,
        template,
        fields,
        config,
        materialization_config=materialization_config,
    )


def render_profile_label(
    graph: Graph,
    root: Node,
    profile: LabelProfileConfiguration,
    config: GraphConfiguration,
    materialization_config: MaterializationConfiguration | None = None,
) -> str | None:
    return render_template(
        graph,
        root,
        profile.template,
        profile.fields,
        config,
        materialization_config=materialization_config,
    )


def label_profile_for_node(
    graph: Graph,
    node: Node,
    config: MaterializationConfiguration,
) -> LabelProfileConfiguration | None:
    for type_node in graph.objects(node, RDF.type):
        if isinstance(type_node, URIRef):
            profile = config.label_profile_for_type(str(type_node))
            if profile is not None:
                return profile
    return None


def display_label(
    graph: Graph,
    root: Node,
    config: GraphConfiguration,
    materialization_config: MaterializationConfiguration | None = None,
    use_target_template: bool = True,
) -> str:
    if materialization_config is not None:
        profile = None
        profile_name = getattr(config, "label_profile", None)
        if use_target_template and profile_name:
            profile = materialization_config.label_profiles.get(profile_name)
        if profile is None:
            profile = label_profile_for_node(
                graph,
                root,
                materialization_config,
            )
        if profile is not None:
            label = render_profile_label(
                graph,
                root,
                profile,
                config,
                materialization_config=materialization_config,
            )
            if label:
                return label

    if use_target_template:
        label = render_label_template(
            graph,
            root,
            config,
            materialization_config=materialization_config,
        )
        if label:
            return label

    label = best_label(graph, root, config)
    if label:
        return normalize_label(label)

    return str(root)


def root_iris(graph: Graph, root_type: str):
    for node in graph.subjects(RDF.type, URIRef(root_type)):
        if isinstance(node, URIRef):
            yield str(node)


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def materialize_records(
    graph: Graph,
    config: MaterializationConfiguration,
    target: str | None = None,
    limit: int | None = None,
) -> list[OutputRecord]:
    target_configs = (
        [config.for_target(target)] if target else list(config.iter_targets())
    )

    by_digest: dict[str, OutputRecord] = {}

    for target_config in target_configs:
        count = 0
        for iri in root_iris(graph, target_config.type):
            if limit is not None and count >= limit:
                break
            count += 1

            node = URIRef(iri)
            label = display_label(
                graph,
                node,
                target_config,
                materialization_config=config,
            )
            text = build_embedding_text(
                graph,
                node,
                target_config,
                label=label,
                materialization_config=config,
            )
            digest = text_digest(text)

            record = by_digest.get(digest)
            if record is None:
                by_digest[digest] = OutputRecord(
                    iris=[iri],
                    label=label,
                    embedding_text=text,
                )
            elif iri not in record.iris:
                record.iris.append(iri)

    records = list(by_digest.values())
    for record in records:
        record.iris.sort()
    return sorted(records, key=lambda r: (r.embedding_text, r.iris))


def write_json(records: Iterable[OutputRecord], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in records],
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_text(records: Iterable[OutputRecord], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(f"label: {r.label}\n")
            f.write("iris:\n")
            for iri in r.iris:
                f.write(f"- {iri}\n")
            f.write("\n")
            f.write(r.embedding_text)
            f.write("\n\n---\n\n")


@app.command()
def materialize_type(hdt_file: Path, target_type: str):
    graph = load_graph(hdt_file)
    config = GraphConfiguration()

    for node_uri in graph.subjects(RDF.type, URIRef(target_type)):
        build_embedding_text(graph, node_uri, config)


@app.command()
def materialize(hdt_file: Path, config_toml: Path):
    graph = load_graph(hdt_file)
    config = MaterializationConfiguration.from_toml(config_toml)

    for target_config in config.iter_targets():
        for node_uri in graph.subjects(RDF.type, URIRef(target_config.type)):
            for level, s, p, o in walk_graph(
                graph, node_uri, target_config
            ):
                print("    " * level, end="")
                print(level, s, p, o)
            break


@app.command()
def materialize_debug(hdt_file: Path, config_toml: Path):
    graph = load_graph(hdt_file)
    config = MaterializationConfiguration.from_toml(config_toml)

    for target_config in config.iter_targets():
        for node_uri in graph.subjects(RDF.type, URIRef(target_config.type)):
            for level, s, p, o in walk_graph(
                graph, node_uri, target_config
            ):
                print("    " * level, end="")
                print(level, s, p, o)
            break


@app.command()
def walk(hdt_file: Path, target_node: str):
    graph = load_graph(hdt_file)

    for p, o in graph.predicate_objects(URIRef(target_node)):
        print(best_label(graph, p, use_humanize=False))
        print(best_label(graph, o, use_humanize=False))
        print()


if __name__ == "__main__":
    app()
