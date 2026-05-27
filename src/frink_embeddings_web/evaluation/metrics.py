"""Shared evaluation/report helper functions.

These functions originally lived in `eval.py` and are intentionally kept
lightweight so both the evaluator and the report generator can reuse them.
"""

from __future__ import annotations

from typing import Any

from qdrant_client.models import ScoredPoint


def get_ids(results: list[ScoredPoint]):
    ids: list[str] = []
    for r in results:
        if r.payload and "iri" in r.payload:
            ids.append(r.payload["iri"])
    return ids


def recall_at_k(a: list[ScoredPoint], b: list[ScoredPoint]):
    """Recall of b vs a.

    Note: this matches the original `eval.py` behavior: it computes overlap / len(a)
    using payload IRIs as identifiers.

    For recall@k, pass in slices (e.g. `knn[:k]`, `ann[:k]`).
    """

    a_ids = set(get_ids(a))
    b_ids = set(get_ids(b))
    overlap = len(a_ids & b_ids)
    return overlap / len(a), overlap


def fmt_range(points: list[ScoredPoint]):
    return f"{points[-1].score:.2f} to {points[0].score:.2f}"


def point_to_row(point: ScoredPoint) -> dict[str, Any]:
    payload = point.payload or {}
    return {
        "score": float(point.score) if point.score is not None else None,
        "iri": payload.get("iri"),
        "label": payload.get("label"),
        "graph": payload.get("graph"),
        "repr": payload.get("repr"),
    }


def top_n_rows(points: list[ScoredPoint], n: int) -> list[dict[str, Any]]:
    return [point_to_row(p) for p in points[:n]]


def missing_points_in_allgraph(
    knn_results: list[ScoredPoint],
    ann_allgraph_results: list[ScoredPoint],
):
    """Implements the previously commented-out `missing_points` logic.

    Returns:
      (missing_points, all_results_min_score)

    A KNN point counts as "missing" if:
      - its score is above the minimum score in the ANN all-graph results
      - its IRI is not present in the ANN all-graph result set
    """

    if not ann_allgraph_results:
        return [], None

    all_results_min = ann_allgraph_results[-1].score
    all_ids = set(get_ids(ann_allgraph_results))

    missing_points = []
    for point in knn_results:
        iri = (point.payload or {}).get("iri")
        if iri is None:
            continue
        if point.score > all_results_min and iri not in all_ids:
            missing_points.append(point)

    return missing_points, float(all_results_min)
