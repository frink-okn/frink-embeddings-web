import hashlib
from typing import Any

from .models import GraphConfiguration

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


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
