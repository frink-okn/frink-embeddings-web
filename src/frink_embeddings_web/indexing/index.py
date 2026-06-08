import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generator, Iterable
from typing import Literal as TypingLiteral

from loguru import logger
from rdflib import BNode, Graph, Literal, Node, URIRef
from rdflib.namespace import RDF
from rdflib.util import from_n3
from rdflib_hdt import HDTStore

from .models import (
    GraphConfiguration,
    LabelProfileConfiguration,
    MaterializationConfiguration,
)

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


@dataclass
class OutputRecord:
    iris: list[str]
    label: str
    embedding_text: str
    # Total distinct IRIs that produced this text, before `iris` is capped at
    # `max_iris_per_record`. May exceed `len(iris)`.
    iri_count: int


def load_graph(hdt_file: Path):
    store = HDTStore(str(hdt_file))
    return Graph(store=store)


# --- graph backends -------------------------------------------------------
#
# A backend reads triples out of the graph as lightweight `GraphTerm`s. The
# HDT backend reads the native HDT document directly, which avoids rdflib's
# per-term object construction -- the dominant cost when materializing.


@dataclass(frozen=True)
class GraphTerm:
    """An RDF term as a `kind` plus its string value.

    Used instead of full rdflib `URIRef`/`Literal` objects on the materialize
    hot path: constructing/validating rdflib terms for every triple is a large
    share of the cost, and we only ever need the string value here.
    """

    kind: TypingLiteral["iri", "literal", "bnode"]
    value: str

    def __str__(self) -> str:
        return self.value


class GraphBackend:
    """Read access to a graph in terms of `GraphTerm`s."""

    def root_iris(self, root_type: str) -> Iterable[str]:
        raise NotImplementedError

    def predicate_objects(
        self, subject: Any
    ) -> Iterable[tuple[GraphTerm, GraphTerm]]:
        raise NotImplementedError

    def objects(self, subject: Any, predicate_iri: str) -> Iterable[GraphTerm]:
        raise NotImplementedError

    def term(self, node: Any) -> GraphTerm:
        raise NotImplementedError

    def types(self, node: Any) -> Iterable[str]:
        for obj in self.objects(node, str(RDF.type)):
            if obj.kind == "iri":
                yield obj.value

    def first_literal(
        self, subject: Any, predicates: Iterable[str]
    ) -> str | None:
        for predicate in predicates:
            for obj in self.objects(subject, predicate):
                if obj.kind == "literal":
                    return obj.value
        return None


class RDFLibBackend(GraphBackend):
    """Backend over any in-memory rdflib graph (used by tests)."""

    def __init__(self, graph: Graph):
        self.graph = graph

    def root_iris(self, root_type: str) -> Iterable[str]:
        for node in self.graph.subjects(RDF.type, URIRef(root_type)):
            if isinstance(node, URIRef):
                yield str(node)

    def predicate_objects(
        self, subject: Any
    ) -> Iterable[tuple[GraphTerm, GraphTerm]]:
        node = self._to_node(subject)
        if isinstance(node, Literal):
            return
        for pred, obj in self.graph.predicate_objects(node):
            if isinstance(pred, URIRef):
                yield self.term(pred), self.term(obj)

    def objects(self, subject: Any, predicate_iri: str) -> Iterable[GraphTerm]:
        node = self._to_node(subject)
        if isinstance(node, Literal):
            return
        for obj in self.graph.objects(node, URIRef(predicate_iri)):
            yield self.term(obj)

    def term(self, node: Any) -> GraphTerm:
        if isinstance(node, GraphTerm):
            return node
        if isinstance(node, Literal):
            return GraphTerm("literal", str(node))
        if isinstance(node, BNode):
            return GraphTerm("bnode", str(node))
        return GraphTerm("iri", str(node))

    def _to_node(self, node: Any) -> Node:
        if isinstance(node, GraphTerm):
            if node.kind == "literal":
                return Literal(node.value)
            if node.kind == "bnode":
                return BNode(node.value)
            return URIRef(node.value)
        return node


class HDTBackend(GraphBackend):
    """Backend that reads the native HDT document, skipping rdflib terms."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.document = graph.store.hdt_document

    def root_iris(self, root_type: str) -> Iterable[str]:
        triples, _ = self.document.search_triples("", str(RDF.type), root_type)
        for subject, _, _ in triples:
            if not subject.startswith("_:"):
                yield subject

    def predicate_objects(
        self, subject: Any
    ) -> Iterable[tuple[GraphTerm, GraphTerm]]:
        subject_term = self.term(subject)
        if subject_term.kind == "literal":
            return
        triples, _ = self.document.search_triples(subject_term.value, "", "")
        for _, pred, obj in triples:
            yield GraphTerm("iri", pred), self._term_from_hdt(obj)

    def objects(self, subject: Any, predicate_iri: str) -> Iterable[GraphTerm]:
        subject_term = self.term(subject)
        if subject_term.kind == "literal":
            return
        triples, _ = self.document.search_triples(
            subject_term.value, predicate_iri, ""
        )
        for _, _, obj in triples:
            yield self._term_from_hdt(obj)

    def term(self, node: Any) -> GraphTerm:
        if isinstance(node, GraphTerm):
            return node
        if isinstance(node, Literal):
            return GraphTerm("literal", str(node))
        if isinstance(node, BNode):
            return GraphTerm("bnode", f"_:{node}")
        if isinstance(node, URIRef):
            return GraphTerm("iri", str(node))
        return self._term_from_hdt(str(node))

    def _term_from_hdt(self, text: str) -> GraphTerm:
        if text.startswith("_:"):
            return GraphTerm("bnode", text)
        if text.startswith('"'):
            try:
                node = from_n3(text)
            except Exception:
                return GraphTerm("literal", text)
            if isinstance(node, Literal):
                return GraphTerm("literal", str(node))
            return GraphTerm("literal", text)
        return GraphTerm("iri", text)


def backend_for_graph(graph: Graph) -> GraphBackend:
    if hasattr(graph.store, "hdt_document"):
        return HDTBackend(graph)
    return RDFLibBackend(graph)


# --- text helpers (pure) --------------------------------------------------


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


def normalize_label(text: str) -> str:
    return " ".join(text.split())


def stable_score(root: Any, pred: Any, obj: Any) -> str:
    text = f"{root}\t{pred}\t{obj}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- materialization ------------------------------------------------------


class GraphReader:
    """Reads and interprets a single graph for materialization.

    Bound to one graph, it owns the backend (rdflib or native HDT) and a
    predicate-label cache, and exposes the operations the textifier needs:
    traversing edges, resolving human-readable labels, and building embedding
    text. Holding that state here keeps it off every call signature. `config`
    is the top-level `MaterializationConfiguration` (for label profiles); the
    per-node methods also take the resolved per-target `config` they apply.
    """

    def __init__(
        self,
        graph: Graph,
        config: MaterializationConfiguration | None = None,
    ):
        self.backend = backend_for_graph(graph)
        self.config = config
        self._predicate_cache: dict[tuple[str, int], str] = {}

    def root_iris(self, root_type: str) -> Iterable[str]:
        return self.backend.root_iris(root_type)

    def best_label(
        self,
        node: Any,
        config: GraphConfiguration | None = None,
        use_fallback: bool = True,
        use_humanize: bool = True,
    ) -> str | None:
        term = self.backend.term(node)

        if term.kind == "literal":
            return term.value

        label_predicates = (
            effective_label_predicates(config)
            if config is not None
            else (RDFS_LABEL,)
        )

        if term.kind != "bnode":
            label = self.backend.first_literal(term, label_predicates)
            if label:
                return label

        if term.kind == "iri" and use_fallback:
            if use_humanize:
                return humanize(iri_fragment(term.value))
            return term.value

        return None

    def predicate_text(self, pred: Any, config: GraphConfiguration) -> str:
        pred_term = self.backend.term(pred)
        key = (pred_term.value, id(config))
        cached = self._predicate_cache.get(key)
        if cached is not None:
            return cached

        label = self.best_label(pred_term, config)
        if not label:
            label = fallback_label(pred_term.value)
        text = humanize(label).lower()

        self._predicate_cache[key] = text
        return text

    def walk(
        self,
        root: Any,
        config: GraphConfiguration,
        expansion_level: int = 0,
    ) -> Generator[tuple[int, GraphTerm, GraphTerm, GraphTerm], None, None]:
        root_term = self.backend.term(root)
        ignore_predicates = set(config.ignore_predicates or [])

        objects_by_predicate: defaultdict[GraphTerm, list[GraphTerm]] = (
            defaultdict(list)
        )
        for p, o in self.backend.predicate_objects(root_term):
            if p.value in ignore_predicates:
                continue
            objects_by_predicate[p].append(o)

        for p in sorted(objects_by_predicate, key=lambda term: term.value):
            objects = objects_by_predicate[p]
            if config.predicate_limit is not None:
                objects = sorted(
                    objects,
                    key=lambda o: stable_score(root_term, p, o),
                )[: config.predicate_limit]

            for o in objects:
                yield expansion_level, root_term, p, o

                if (
                    o.kind != "literal"
                    and expansion_level < config.expansion_limit
                ):
                    yield from self.walk(o, config, expansion_level + 1)

    def first_direct_value(
        self,
        root: Any,
        predicate_iri: str,
        config: GraphConfiguration,
    ) -> str | None:
        values = []
        for obj in self.backend.objects(root, predicate_iri):
            if obj.kind == "bnode":
                continue
            label = self.display_label(obj, config, use_target_template=False)
            if label:
                values.append(label)

        if not values:
            return None
        return sorted(values)[0]

    def render_template(
        self,
        root: Any,
        template: str,
        fields: dict[str, str],
        config: GraphConfiguration,
    ) -> str | None:
        def replace(match: re.Match[str]) -> str:
            field = match.group(1).strip()
            predicate_iri = fields.get(field)
            if predicate_iri is None:
                return ""
            return self.first_direct_value(root, predicate_iri, config) or ""

        label = re.sub(r"\{([^{}]+)\}", replace, template)
        return normalize_label(label) or None

    def render_label_template(
        self, root: Any, config: GraphConfiguration
    ) -> str | None:
        template = getattr(config, "label_template", None)
        fields = getattr(config, "label_fields", {})
        if not template:
            return None
        return self.render_template(root, template, fields, config)

    def render_profile_label(
        self,
        root: Any,
        profile: LabelProfileConfiguration,
        config: GraphConfiguration,
    ) -> str | None:
        return self.render_template(
            root, profile.template, profile.fields, config
        )

    def label_profile_for_node(
        self, node: Any
    ) -> LabelProfileConfiguration | None:
        if self.config is None or not self.config.label_profiles:
            return None
        for type_iri in self.backend.types(node):
            profile = self.config.label_profile_for_type(type_iri)
            if profile is not None:
                return profile
        return None

    def display_label(
        self,
        root: Any,
        config: GraphConfiguration,
        use_target_template: bool = True,
    ) -> str:
        root_term = self.backend.term(root)

        if self.config is not None:
            profile = None
            profile_name = getattr(config, "label_profile", None)
            if use_target_template and profile_name:
                profile = self.config.label_profiles.get(profile_name)
            if profile is None and self.config.label_profiles:
                profile = self.label_profile_for_node(root_term)
            if profile is not None:
                label = self.render_profile_label(root_term, profile, config)
                if label:
                    return label

        if use_target_template:
            label = self.render_label_template(root_term, config)
            if label:
                return label

        label = self.best_label(root_term, config)
        if label:
            return normalize_label(label)

        return root_term.value

    def build_embedding_text(
        self,
        root: Any,
        config: GraphConfiguration,
        label: str | None = None,
    ) -> str:
        root_term = self.backend.term(root)
        lines: list[str] = []

        if label is None:
            label = self.display_label(root_term, config)
        if label:
            lines.append(f"label: {label}")

        for level, _, p, o in self.walk(root_term, config):
            if level > 0:
                continue
            if p.kind != "iri":
                continue

            pred_txt = self.predicate_text(p, config)
            obj_txt = self.display_label(o, config, use_target_template=False)

            if not obj_txt or o.kind == "bnode":
                continue

            lines.append(f"{pred_txt}: {obj_txt}")

        return "\n".join(lines)


def materialize_records(
    graph: Graph,
    config: MaterializationConfiguration,
    target: str | None = None,
    limit: int | None = None,
    max_iris_per_record: int = 10,
) -> list[OutputRecord]:
    reader = GraphReader(graph, config)
    target_configs = (
        [config.for_target(target)] if target else list(config.iter_targets())
    )

    by_digest: dict[str, OutputRecord] = {}

    for target_config in target_configs:
        count = 0
        for iri in reader.root_iris(target_config.type):
            if limit is not None and count >= limit:
                break
            count += 1

            node = GraphTerm("iri", iri)
            label = reader.display_label(node, target_config)
            text = reader.build_embedding_text(node, target_config, label=label)
            digest = text_digest(text)

            record = by_digest.get(digest)
            if record is None:
                by_digest[digest] = OutputRecord(
                    iris=[iri],
                    label=label,
                    embedding_text=text,
                    iri_count=1,
                )
            elif iri not in record.iris:
                record.iri_count += 1
                if len(record.iris) < max_iris_per_record:
                    record.iris.append(iri)

    records = list(by_digest.values())
    for record in records:
        record.iris.sort()
        if record.iri_count > max_iris_per_record:
            logger.warning(
                "Truncated grouped record IRIs from {} to {} for "
                "embedding text:\n{}",
                record.iri_count,
                len(record.iris),
                record.embedding_text,
            )
    return sorted(records, key=lambda r: (r.embedding_text, r.iris))


# --- output writers -------------------------------------------------------


def write_json(records: Iterable[OutputRecord], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in records],
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_jsonl(records: Iterable[OutputRecord], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            write_jsonl_record(record, f)


def write_jsonl_record(record: OutputRecord, f) -> None:
    json.dump(asdict(record), f, ensure_ascii=False)
    f.write("\n")


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
