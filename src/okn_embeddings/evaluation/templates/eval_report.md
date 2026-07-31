# Qdrant query report

## HNSW settings

* `HNSW_M` (index time): {{ collection_settings.config.hnsw_config.m }}
* `HNSW_EF_CONSTRUCT` (index time): {{ collection_settings.config.hnsw_config.ef_construct }}
* `HNSW_EF` (query time): {{ hnsw_ef }}


## Methodology

This report compares three retrieval modes (cosine similarity) against the same Qdrant collection:

* **Exact KNN**: Qdrant `query_points(..., exact=True)` with the query restricted to the target graph.
* **In-graph ANN**: Qdrant HNSW ANN with the same target-graph filter.
* **All-graph ANN**: Qdrant HNSW ANN with no graph filter (the entire collection is the candidate set).

Notes:
* `recall@k (in-graph ANN vs exact KNN)` is primarily measuring ANN approximation quality within the *same* candidate set.
* `recall@k (all-graph ANN vs exact KNN)` uses a *different* candidate set (global vs in-graph), so interpret it as “do the in-graph exact neighbors also show up in global retrieval?”, not strictly “ANN error”.


## Overall summary

* Queries: {{ overall_summary.num_queries }}
* Mean missing-in-all-graph count: {{ "%.2f"|format(overall_summary.mean_missing_in_allgraph) }}
* Max missing-in-all-graph count: {{ overall_summary.max_missing_in_allgraph }}

### Mean recall@k (vs exact KNN)

| k | in-graph ANN | all-graph ANN |
|---:|---:|---:|
{% for row in overall_summary.mean_recall %}
| {{ row.k }} | {{ "%.3f"|format(row.in_graph_recall) }} | {{ "%.3f"|format(row.all_graph_recall) }} |
{% endfor %}


## Queries

{% for graph, graph_queries in query_contexts|groupby("target_graph") %}
### {{ graph }}

#### Graph summary

* Queries: {{ graph_summaries[graph].num_queries }}
* Mean missing-in-all-graph count: {{ "%.2f"|format(graph_summaries[graph].mean_missing_in_allgraph) }}
* Max missing-in-all-graph count: {{ graph_summaries[graph].max_missing_in_allgraph }}

| k | in-graph ANN | all-graph ANN |
|---:|---:|---:|
{% for row in graph_summaries[graph].mean_recall %}
| {{ row.k }} | {{ "%.3f"|format(row.in_graph_recall) }} | {{ "%.3f"|format(row.all_graph_recall) }} |
{% endfor %}

{% for q in graph_queries %}
#### Query: '{{ q.query }}'

**Recall@k (vs exact KNN)**

| k | in-graph recall | in-graph overlap | all-graph recall | all-graph overlap |
|---:|---:|---:|---:|---:|
{% for k in q.ks %}
| {{ k }} | {{ "%.3f"|format(q.recall[k].in_graph.recall) }} | {{ q.recall[k].in_graph.overlap }} | {{ "%.3f"|format(q.recall[k].all_graph.recall) }} | {{ q.recall[k].all_graph.overlap }} |
{% endfor %}

**Cosine similarity ranges**

* KNN: {{ q.score_ranges.knn }}
* ANN (in-graph): {{ q.score_ranges.ann_ingraph }}
* ANN (all-graph): {{ q.score_ranges.ann_allgraph }}

**Top 5 results**

*Exact KNN*

| rank | score | iri | label |
|---:|---:|---|---|
{% for r in q.top5.knn %}
| {{ loop.index }} | {{ "%.4f"|format(r.score) }} | {{ r.iri }} | {{ r.label }} |
{% endfor %}

*ANN (in-graph)*

| rank | score | iri | label |
|---:|---:|---|---|
{% for r in q.top5.ann_ingraph %}
| {{ loop.index }} | {{ "%.4f"|format(r.score) }} | {{ r.iri }} | {{ r.label }} |
{% endfor %}

*ANN (all-graph)*

| rank | score | iri | label |
|---:|---:|---|---|
{% for r in q.top5.ann_allgraph %}
| {{ loop.index }} | {{ "%.4f"|format(r.score) }} | {{ r.iri }} | {{ r.label }} |
{% endfor %}

**Missed points in full ANN**

* all-graph ANN min score: {{ "%.4f"|format(q.missing_in_allgraph.all_results_min) if q.missing_in_allgraph.all_results_min is not none else "n/a" }}
* missed count: {{ q.missing_in_allgraph.count }}

{% if q.missing_in_allgraph.top10|length > 0 %}
Top 10 missed (by KNN score):

| rank | score | iri | label |
|---:|---:|---|---|
{% for r in q.missing_in_allgraph.top10 %}
| {{ loop.index }} | {{ "%.4f"|format(r.score) }} | {{ r.iri }} | {{ r.label }} |
{% endfor %}
{% endif %}

{% endfor %}

{% endfor %}
