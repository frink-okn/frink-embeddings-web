import re
from collections import defaultdict
from typing import Any, Generator

from loguru import logger
from rdflib import Graph

from .models import (
    GraphConfiguration,
    LabelProfileConfiguration,
    MaterializationConfiguration,
)
from .output import OutputRecord
from .reader import GraphReader, GraphTerm, graph_reader
from .text import (
    RDFS_LABEL,
    effective_label_predicates,
    fallback_label,
    humanize,
    iri_fragment,
    normalize_label,
    stable_score,
    text_digest,
)


class Textifier:
    """Turns a graph's nodes into labels and embedding text.

    Holds a `GraphReader` (the read layer) plus the top-level
    `MaterializationConfiguration` (for label profiles) and a predicate-label
    cache, so the walk/label/text helpers don't thread that state through every
    call. The per-node methods take the resolved per-target `config` they apply.
    """

    def __init__(
        self,
        reader: GraphReader,
        config: MaterializationConfiguration | None = None,
    ):
        self.reader = reader
        self.config = config
        self._predicate_cache: dict[tuple[str, int], str] = {}

    def best_label(
        self,
        node: Any,
        config: GraphConfiguration | None = None,
        use_fallback: bool = True,
        use_humanize: bool = True,
    ) -> str | None:
        term = self.reader.term(node)

        if term.kind == "literal":
            return term.value

        label_predicates = (
            effective_label_predicates(config)
            if config is not None
            else (RDFS_LABEL,)
        )

        if term.kind != "bnode":
            label = self.reader.first_literal(term, label_predicates)
            if label:
                return label

        if term.kind == "iri" and use_fallback:
            if use_humanize:
                return humanize(iri_fragment(term.value))
            return term.value

        return None

    def predicate_text(self, pred: Any, config: GraphConfiguration) -> str:
        pred_term = self.reader.term(pred)
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
        root_term = self.reader.term(root)
        ignore_predicates = set(config.ignore_predicates or [])

        objects_by_predicate: defaultdict[GraphTerm, list[GraphTerm]] = (
            defaultdict(list)
        )
        for p, o in self.reader.predicate_objects(root_term):
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
        for obj in self.reader.objects(root, predicate_iri):
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
        for type_iri in self.reader.types(node):
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
        root_term = self.reader.term(root)

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
        root_term = self.reader.term(root)
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
    graph: "Graph",
    config: MaterializationConfiguration,
    target: str | None = None,
    limit: int | None = None,
    max_iris_per_record: int = 10,
) -> list[OutputRecord]:
    reader = graph_reader(graph)
    textifier = Textifier(reader, config)
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
            label = textifier.display_label(node, target_config)
            text = textifier.build_embedding_text(
                node, target_config, label=label
            )
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
