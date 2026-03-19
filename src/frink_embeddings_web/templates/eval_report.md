# Qdrant query report

## HNSW settings

* `HNSW_M` (index time): {{ collection_settings.config.hnsw_config.m }}
* `HNSW_EF_CONSTRUCT` (index time): {{ collection_settings.config.hnsw_config.ef_construct }}
* `HNSW_EF` (query time): {{ hnsw_ef }}


## Queries

{% for graph, graph_queries in queries|groupby("target_graph") %}
### {{ graph }}

{% for query in graph_queries %}
#### Query: '{{ query.query }}'
{% endfor %}

{% endfor %}
