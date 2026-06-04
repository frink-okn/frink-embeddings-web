---
name: query-okn
description: Query the Open Knowledge Network (OKN) federated knowledge graph using text embeddings and graph-based RDF query utilities
---

Use this skill when asked to find information contained in the Open Knowledge Network (OKN), a federation of independently published RDF knowledge graphs spanning many domains (biomedical, geoscience, supply chain, hydrology, and more). Each graph is a distinct dataset; a question may be answerable from one graph, several, or none.

The general approach: use the **text embeddings** to *discover* where relevant entities live, then use **RDF queries** (SPARQL / Triple Pattern Fragments) against those graphs to *retrieve precise facts*.

## About the text embeddings

The text embeddings aren't a full picture of what is in the graph. They are a curated set of targets, whose text materializations are made up of walking the graph and getting text representations of a certain set of predicates. Do not assume that an embedding search is complete.

Concretely, this means:

- **Only some nodes are embedded.** Indexing picks certain RDF *types* as root "targets"; nodes of other types are not directly searchable even though they exist in the graph. (See the `textify` skill for how this materialization works.)
- **The embedded text is a label-like summary**, built from a selected set of predicates — not the full content of the node.
- **Identical materialized text is deduplicated.** One result can therefore stand for many IRIs: each hit carries a representative `primary_uri` plus the full `iris` list and an `iri_count`.
- **A miss is not evidence of absence.** If something isn't found via embeddings, it may still be in the graph — confirm with an RDF query before concluding it isn't there.

Treat embedding search as a fuzzy entry point, not a source of truth.

## CLI Utilities

You have access to several utilities for querying text embeddings from the OKN knowledge graph. These are all available under the `frink-search` command line utility, which you will run using `uv run frink-search`. Pass `--json` to any command so the output is machine-readable.

### Search

```
uv run frink-search search <TERM> [options]
```

Returns the closest embedding matches across all graphs (or a chosen subset), best match first.

- `-t, --type text|node` — embed `<TERM>` as free text (default), or treat `<TERM>` as an existing node IRI and find embeddings *similar to that node* ("find similar").
- `-g, --graph <GRAPH>` — restrict to one or more graphs (repeatable). `-x, --exclude-graph <GRAPH>` excludes them instead (the two are mutually exclusive).
- `-l, --limit <N>` — number of results (default 10); `--offset <N>` to page through.
- `--show-repr` — include each hit's materialized embedding text, so you can judge *why* it matched.
- `--json` — emit JSON. Each result has: `score` (cosine similarity, higher is closer), `label`, `graph` (the source graph — this is the `{GRAPH}` slug for the RDF endpoints below), `primary_uri` (the representative IRI — use this for RDF follow-up), `iris` (all IRIs sharing this embedding), and `iri_count`.

### Survey

```
uv run frink-search survey <TERM> [options]
```

Like `search`, but queries every graph independently and returns the top matches *per graph* — so you can see where across the federation a concept appears. Accepts the same `-t/-g/-x/--show-repr/--json` options; here `-l, --limit <N>` is the number of results **per graph** (default 5). JSON output is an array of `{ "graph": ..., "results": [ ... ] }`, ordered with the best-matching graphs first.

### List graphs

```
uv run frink-search list-graphs [--sort name|count] [--json]
```

Lists the graphs in the collection and how many embeddings each contains. Useful to see the universe of available graphs before narrowing with `-g` / `-x`.

### Strategies

1. **Orient.** Start with a `survey` of your term to see which graphs are most relevant (or `list-graphs` to see what exists at all).
2. **Dig in.** Once you know the promising graphs, run `search` restricted to them (`-g <graph>`) with a larger `--limit` to gather more candidate entities.
3. **Expand (optional).** Given a good node IRI, `search --type node <IRI>` pulls semantically similar nodes — another way to broaden beyond the literal term.
4. **Fan out across graphs (optional).** If you've found a highly relevant node and want to see whether *other* graphs hold similar nodes, re-run `search --type node <IRI>` while excluding the graph you found it in (`-x <graph>`). If those results then saturate with terms from another single graph, exclude that one too and search again — and keep going. Continue this fan-out until excluding graphs stops surfacing new relevant terms, at which point you can be confident you've covered the relevant material across the federation.
5. **Switch to RDF.** Because the embeddings are imprecise and incomplete, treat the discovered IRIs as a *starting point* and move to RDF queries for precise answers.

Each graph exposes both a SPARQL endpoint and a Triple Pattern Fragments (TPF) endpoint, where `{GRAPH}` is the `graph` value from a search/survey result:

- `https://apps.okn.us/{GRAPH}/sparql`
- `https://apps.okn.us/ldf/{GRAPH}`

**Prefer TPF for simple lookups and walks** — it is much lighter weight on the servers than SPARQL. Walk out from a URI you found via embedding search by querying it as both subject and object to get its out-edges and in-edges:

- `<URI> ?p ?o` — what this entity points to
- `?s ?p <URI>` — what points to this entity

Only reach for **SPARQL when the question genuinely needs it** — joins across multiple entities, filtering, aggregation/counting, or multi-hop patterns that would otherwise take many TPF requests. Keep such queries targeted.

### Notes

Keep a running count of how many queries you made to the text embedding utilities, and how many TPF and SPARQL queries you made. Report them back at the end.
