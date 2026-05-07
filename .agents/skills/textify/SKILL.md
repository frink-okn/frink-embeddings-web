---
name: frink-rdf-textify
description: Convert RDF nodes into textual representations for the purpose of creating text embeddings
---

Use this skill when asked to design and evaluate an indexing configuration to be used with the `frink-indexing textify` command to create text embeddings from RDF graphs.

The overall purpose is to generate textual representations of selected RDF nodes so they can be embedded with a text embedding model. RDF graphs are usually too broad and heterogeneous to embed every node directly. A good indexing configuration chooses which RDF types should become embedding documents, how those documents should be labeled, which graph edges should contribute text, and which noisy or structural graph regions should be skipped.

At a high level, the `frink-indexing` textification algorithm starts from configured RDF types, finds subject IRIs of those types, assigns each root node a human-readable label, walks selected outgoing graph edges, and turns predicate/object pairs into compact text lines. Object IRIs are represented by derived labels rather than full nested graph dumps. Label profiles can be defined separately from textification targets, so helper classes can provide good labels when they appear during graph walking without becoming standalone embedding documents. The output records group equivalent text under one record, preserve source IRIs, and include both a display label and embedding text.

The end product is a concise, evidence-backed `config.toml` for `frink-indexing textify`, plus a report explaining the choices.

## CLI Utilities

The `frink-indexing` CLI is used to explore an HDT graph, evaluate candidate configurations, and produce final textification records. You may not have a virtual environment activated, so prefix any CLI invocation of this or other Python commands with `uv run`.

- `frink-indexing sample-types`: graph exploration. Use this before drafting a config. It scans RDF types, counts instances, samples subject IRIs, reports direct literal predicates, and reports outgoing object predicates with object type and label evidence. This helps identify good root targets, possible label predicates or label profiles, and noisy predicates to ignore.
- `frink-indexing sample-targets`: config evaluation. Use this after drafting or editing a config. It runs the configured target definitions with a small per-target limit and writes representative sample documents. This is the main tool for judging whether the config produces compact, readable, semantically useful embedding text.
- `frink-indexing textify`: final or bounded materialization. Use this to generate the actual JSON/text records for embedding. During config design, run it with a small `--limit`; for production, run it without a limit after the config has been reviewed.

## Workflow

1. If not provided, ask for a graph to run this skill against. Currently the library only supports HDT files.
2. Create a working folder in the same directory as the graph called `${GRAPH_NAME}-textify`.
3. Run `frink-indexing sample-types` on the HDT graph to understand graph shape.
    - Use small per-type limits first, for example `--limit 2 --values-limit 2`.
    - Prefer JSON for machine review and text for human inspection.
4. Identify candidate root types.
    - Prefer entity-like types that produce compact, meaningful documents.
    - Avoid structural/helper types, extremely numerous measurement/value nodes, geometries, units, blank-node-like containers, and classes that only have ontology statements (although, of course, include ontology statements if the whole graph is an OWL ontology).
    - Record each type count and what the type appears to represent.
5. For each candidate type, inspect:
    - direct literal predicates that can become labels or useful text.
    - outgoing object predicates that lead to useful labeled entities.
    - object type distributions and object label predicates.
    - high-fanout or noisy predicates to ignore.
6. Draft `config.toml`.
    - Look at the file `indexing/models.py` for information about what a graph configuration consists of. You can also review `indexing/index.py` to understand the indexing algorithm.
    - Set target `type` values explicitly.
    - Add `label_predicates` that are supported by the sampled data.
    - Prefer `label_predicates` when a graph already has appropriate RDF predicates for names, titles, labels, or descriptions.
    - Only use `label_template`, `label_fields`, or `label_profiles` when no appropriate RDF label predicate exists, or when direct labels are too weak, too generic, or too verbose.
    - Do not create boilerplate label profiles that simply map `{name}` to a normal RDF name predicate; put that predicate in `label_predicates` instead.
    - Use `label_profiles` when helper classes need derived labels during graph walking without becoming textification targets.
    - Set conservative `predicate_limit` and `expansion_limit`; prefer concise embedding text for `all-MiniLM-L6-v2`.
    - Add `ignore_predicates` for noisy, huge, structural, or low-semantic-value predicates.
7. Run `frink-indexing sample-targets` and, when useful, `frink-indexing textify` with small `--limit` values.
8. Iterate until the sampled embedding text is compact, readable, and semantically useful.
9. Write a report in the folder with the rest of the output titled `README.md`.
    - commands run.
    - included target types and type counts.
    - one summary line for every targeted class describing what the class appears to represent and how many instances exist.
    - skipped types and why.
    - For each targeted type, include:
        * a summary of what this targeted class seems to represent.
        * a count of how many nodes of this type are in the graph.
        * label strategy.
        * ignored predicates and why.
        * sample embedding texts.
    - Then write:
        * a wrap-up assessment that states which representations look strongest, which look weak or noisy, which targets should be kept or reconsidered, and the highest-value next iterations.
        * remaining uncertainties or recommended next iterations.

## Quality Bar

- The config should not try to embed everything.
- Each target should correspond to a thing a user may reasonably search for.
- Embedding text should be concise enough for a small sentence-transformer model.
- Label choices should be stable and human-scannable in a browsing interface.
- Every non-obvious config choice should be backed by sampler evidence.
- Every target in the config should be represented in the report by a rationale, an instance count, a short textual class summary, and at least one sample document.
- The sample-document section should be self-contained: a reader should not need to scroll back to the target table to learn what each sampled class represents.
- The report should end with an opinionated wrap-up based on the sampled documents, not just a neutral inventory.
