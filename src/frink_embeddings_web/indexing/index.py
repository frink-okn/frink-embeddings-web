from collections import Counter
from pathlib import Path
from typing import Any, Generator, Iterable

import typer
from rdflib import Graph, Literal, Node, URIRef
from rdflib.namespace import RDF
from rdflib_hdt import HDTStore

from .models import MaterializationConfiguration

LABEL_PREDICATES = ("http://www.w3.org/2000/01/rdf-schema#label",)

app = typer.Typer()


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


def first_literal(graph: Graph, s: Node, ps: Iterable[str]):
    for p in ps:
        for o in graph.objects(s, URIRef(p)):
            if isinstance(o, Literal):
                return str(o)
    return None


def best_label(graph: Graph, node: Any, use_fallback=True, use_humanize=True):
    if isinstance(node, Literal):
        return str(node)

    if isinstance(node, Node):
        label = first_literal(graph, node, LABEL_PREDICATES)
        if label:
            return label

    if isinstance(node, URIRef) and use_fallback:
        return humanize(iri_fragment(node)) if use_humanize else str(node)

    return None


def walk_graph(
    graph: Graph,
    root: Node,
    *,
    expansion_level=0,
    expansion_limit=1,
    predicate_limit: int | None = None,
    ignore_predicates: Iterable[str] | None = None,
) -> Generator[tuple[int, Node, Node, Node], None, None]:
    p_counts = Counter[str]()

    if ignore_predicates is None:
        ignore_predicates = set()
    else:
        ignore_predicates = set(ignore_predicates)

    for p, o in graph.predicate_objects(root):
        p_str = str(p)
        if p_str in ignore_predicates:
            continue
        p_count = p_counts.get(p_str, 0)
        if predicate_limit is not None and p_count >= predicate_limit:
            continue
        p_counts[p_str] += 1

        yield expansion_level, root, p, o

        if not isinstance(o, Literal) and expansion_level < expansion_limit:
            yield from walk_graph(
                graph,
                o,
                expansion_level=expansion_level + 1,
                expansion_limit=expansion_limit,
                predicate_limit=predicate_limit,
                ignore_predicates=ignore_predicates,
            )


@app.command()
def materialize_type(hdt_file: Path, target_type: str):
    graph = load_graph(hdt_file)

    for node_uri in graph.subjects(RDF.type, URIRef(target_type)):
        build_embedding_text(graph, node_uri)


@app.command()
def materialize(hdt_file: Path, config_toml: Path):
    graph = load_graph(hdt_file)
    config = MaterializationConfiguration.from_toml(config_toml)

    for target_config in config.iter_targets():
        for node_uri in graph.subjects(RDF.type, URIRef(target_config.type)):
            for level, s, p, o in walk_graph(
                graph, node_uri, expansion_limit=2
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
