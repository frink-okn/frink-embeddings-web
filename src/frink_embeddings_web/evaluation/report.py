import json
from pathlib import Path

from jinja2 import Environment, PackageLoader
from pydantic import BaseModel

from ..core.models import TimedQueryResponse
from .eval import Evaluation
from .metrics import (
    fmt_range,
    missing_points_in_allgraph,
    recall_at_k,
    top_n_rows,
)


class RecallRow(BaseModel):
    k: int
    in_graph_recall: float
    in_graph_overlap: int
    all_graph_recall: float
    all_graph_overlap: int


def _recall_table(
    knn_results: TimedQueryResponse,
    ann_ingraph_results: TimedQueryResponse,
    ann_allgraph_results: TimedQueryResponse,
    ks: tuple[int, ...],
):
    rows: list[RecallRow] = []
    for k in ks:
        ref = knn_results.points[:k]

        in_graph_recall, in_graph_overlap = recall_at_k(
            ref, ann_ingraph_results.points[:k]
        )
        all_graph_recall, all_graph_overlap = recall_at_k(
            ref, ann_allgraph_results.points[:k]
        )
        rows.append(
            RecallRow(
                k=k,
                in_graph_recall=in_graph_recall,
                in_graph_overlap=in_graph_overlap,
                all_graph_recall=all_graph_recall,
                all_graph_overlap=all_graph_overlap,
            )
        )
    return rows


def _mean(vals: list[float]):
    return sum(vals) / len(vals) if vals else 0.0


def _summary_from_queries(query_contexts, ks: tuple[int, ...]):
    # mean recall@k across all queries
    mean_recall = []
    for k in ks:
        ig_vals = [
            qc["recall"][k]["in_graph"]["recall"] for qc in query_contexts
        ]
        ag_vals = [
            qc["recall"][k]["all_graph"]["recall"] for qc in query_contexts
        ]
        mean_recall.append(
            {
                "k": k,
                "in_graph_recall": _mean(ig_vals),
                "all_graph_recall": _mean(ag_vals),
            }
        )

    missing_counts = [
        qc["missing_in_allgraph"]["count"] for qc in query_contexts
    ]
    return {
        "num_queries": len(query_contexts),
        "mean_recall": mean_recall,
        "mean_missing_in_allgraph": _mean([float(x) for x in missing_counts]),
        "max_missing_in_allgraph": max(missing_counts) if missing_counts else 0,
    }


def render_report(input_json: Path):
    env = Environment(
        loader=PackageLoader("frink_embeddings_web"),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template("eval_report.md")

    with input_json.open() as fp:
        data = json.load(fp)
        evaluation = Evaluation.model_validate(data)

    ks = (1, 5, 10, 20, 50)

    # Pre-compute everything in Python (keeps the Jinja template simple)
    query_contexts = []
    for q in evaluation.queries:
        recall_rows = _recall_table(
            q.knn_results, q.ann_ingraph_results, q.ann_allgraph_results, ks
        )

        recall_map = {
            r.k: {
                "in_graph": {
                    "recall": float(r.in_graph_recall),
                    "overlap": int(r.in_graph_overlap),
                },
                "all_graph": {
                    "recall": float(r.all_graph_recall),
                    "overlap": int(r.all_graph_overlap),
                },
            }
            for r in recall_rows
        }

        missing_points, all_min = missing_points_in_allgraph(
            q.knn_results.points, q.ann_allgraph_results.points
        )

        query_contexts.append(
            {
                "target_graph": q.target_graph,
                "query": q.query,
                "ks": ks,
                "recall": recall_map,
                "score_ranges": {
                    "knn": fmt_range(q.knn_results.points),
                    "ann_ingraph": fmt_range(q.ann_ingraph_results.points),
                    "ann_allgraph": fmt_range(q.ann_allgraph_results.points),
                },
                "top5": {
                    "knn": top_n_rows(q.knn_results.points, 5),
                    "ann_ingraph": top_n_rows(q.ann_ingraph_results.points, 5),
                    "ann_allgraph": top_n_rows(
                        q.ann_allgraph_results.points, 5
                    ),
                },
                "missing_in_allgraph": {
                    "all_results_min": all_min,
                    "count": len(missing_points),
                    "top10": top_n_rows(missing_points, 10),
                },
            }
        )

    overall_summary = _summary_from_queries(query_contexts, ks)

    graphs = {}
    for qc in query_contexts:
        graphs.setdefault(qc["target_graph"], []).append(qc)

    graph_summaries = {
        graph: _summary_from_queries(qcs, ks) for graph, qcs in graphs.items()
    }

    context = {
        **evaluation.model_dump(),
        "query_contexts": query_contexts,
        "overall_summary": overall_summary,
        "graph_summaries": graph_summaries,
    }

    print(template.render(**context))


if __name__ == "__main__":
    render_report(Path("output.json"))
