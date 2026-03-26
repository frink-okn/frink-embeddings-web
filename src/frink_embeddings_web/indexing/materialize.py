import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import typer
from pyoxigraph import BlankNode, Literal, NamedNode, RdfFormat, Store

app = typer.Typer(add_completion=False)


ROOT_QUERY = """
SELECT DISTINCT ?root
WHERE {{
  ?root a <{root_type}> .
  FILTER(isIRI(?root))
}}
ORDER BY ?root
"""


LABEL_PREDICATES = (
    "http://purl.org/dc/elements/1.1/title",
    "http://purl.org/dc/terms/title",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://schema.org/name",
    "http://xmlns.com/foaf/0.1/name",
    "http://purl.org/ontology/bibo/shortTitle",
    "http://sail.ua.edu/ruralkg/treatment/name",
    "http://sail.ua.edu/ruralkg/administrativearea/name",
)

PERSON_GIVEN = "http://xmlns.com/foaf/0.1/givenname"
PERSON_SURNAME = "http://xmlns.com/foaf/0.1/surname"

SKIP_PREDICATES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#first",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest",
    "https://idir.uta.edu/sockg-ontology#hasMeasurement",
}

LITERAL_EDGE_PREDICATES = {
    "http://purl.org/dc/elements/1.1/date",
    "http://purl.org/dc/terms/date",
    "http://purl.org/ontology/bibo/doi",
    "http://purl.org/ontology/bibo/isbn13",
}


@dataclass
class OutputRecord:
    iri: str
    embedding_text: str


def guess_format(path: Path) -> RdfFormat:
    ext = path.suffix.lower()
    if ext == ".ttl":
        return RdfFormat.TURTLE
    if ext == ".nt":
        return RdfFormat.N_TRIPLES
    if ext == ".nq":
        return RdfFormat.N_QUADS
    if ext == ".trig":
        return RdfFormat.TRIG
    if ext in (".rdf", ".xml"):
        return RdfFormat.RDF_XML
    if ext == ".jsonld":
        return RdfFormat.JSON_LD
    raise typer.BadParameter(f"Unsupported RDF extension: {ext}")


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


def first_literal(store: Store, subject: NamedNode, predicates: Iterable[str]):
    for p in predicates:
        pred = NamedNode(p)
        for q in store.quads_for_pattern(subject, pred, None, None):
            if isinstance(q.object, Literal):
                return q.object.value
    return None


def person_label(store: Store, person: NamedNode, use_fallback=True):
    given = first_literal(store, person, (PERSON_GIVEN,))
    surname = first_literal(store, person, (PERSON_SURNAME,))

    if given and surname:
        return f"{given} {surname}"
    if surname:
        return surname
    if given:
        return given

    return fallback_label(person.value) if use_fallback else None


def best_label(store: Store, node, use_fallback=True):
    if isinstance(node, Literal):
        return node.value

    if isinstance(node, BlankNode):
        return None

    if isinstance(node, NamedNode):
        label = first_literal(store, node, LABEL_PREDICATES)
        if label:
            return label

        person = person_label(store, node, use_fallback)
        if person:
            return person

        return fallback_label(node.value) if use_fallback else None

    return None


def predicate_text(store: Store, pred: NamedNode):
    label = best_label(store, pred)
    if not label:
        label = fallback_label(pred.value)
    return humanize(label).lower()


MAX_PRED_COUNT = 5


def stable_score(root_iri: str, pred_iri: str, obj) -> str:
    if isinstance(obj, Literal):
        obj_key = obj.value
    elif isinstance(obj, NamedNode):
        obj_key = obj.value
    else:
        obj_key = str(obj)

    text = f"{root_iri}\t{pred_iri}\t{obj_key}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_embedding_text(store: Store, root: NamedNode) -> str:
    lines: list[str] = []

    label = best_label(store, root, False)
    if label:
        lines.append(f"entity: {label}")

    quads_by_pred: defaultdict[NamedNode, list] = defaultdict(list)

    for q in store.quads_for_pattern(root, None, None, None):
        pred = q.predicate
        obj = q.object

        if pred.value in SKIP_PREDICATES:
            continue

        if isinstance(obj, BlankNode):
            continue

        quads_by_pred[pred].append(obj)

    selected_by_pred: dict[NamedNode, list] = {}

    for pred, objs in quads_by_pred.items():
        if len(objs) > MAX_PRED_COUNT:
            objs = sorted(
                objs,
                key=lambda obj: stable_score(root.value, pred.value, obj),
            )[:MAX_PRED_COUNT]
        selected_by_pred[pred] = objs

    for pred in sorted(selected_by_pred, key=lambda p: p.value):
        pred_txt = predicate_text(store, pred)

        for obj in selected_by_pred[pred]:
            if isinstance(obj, Literal):
                lines.append(f"{pred_txt}: {obj.value}")
                continue

            obj_label = best_label(store, obj)
            if obj_label:
                lines.append(f"{pred_txt}: {obj_label}")

    return "\n".join(lines)


def load_store(path: Path):
    store = Store()
    fmt = guess_format(path)

    with path.open("rb") as f:
        store.bulk_load(f, format=fmt)

    store.optimize()
    return store


def root_iris(store: Store, root_type: str):
    query = ROOT_QUERY.format(root_type=root_type)
    for row in store.query(query):
        node = row["root"]
        if isinstance(node, NamedNode):
            yield node.value


def write_json(records, output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)


def write_text(records, output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(f"iri: {r.iri}\n")
            f.write(r.embedding_text)
            f.write("\n\n---\n\n")


@app.command()
def materialize(
    rdf_file: Path = typer.Argument(...),
    root_type: str = typer.Option(...),
    output: Path = typer.Option(..., "-o"),
    text: bool = typer.Option(False, help="Output debug text instead of JSON"),
):
    store = load_store(rdf_file)

    records = []
    seen: set[int] = set()

    for iri in root_iris(store, root_type):
        node = NamedNode(iri)
        text_block = build_embedding_text(store, node)
        text_hash = hash(text_block)
        if text_hash in seen:
            continue
        seen.add(text_hash)
        records.append(OutputRecord(iri=iri, embedding_text=text_block))

    if text:
        write_text(records, output)
    else:
        write_json(records, output)

    typer.echo(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    app()
