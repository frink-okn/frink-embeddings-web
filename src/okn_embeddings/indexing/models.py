"""Configuration models for RDF-to-text materialization.

The indexing CLI reads a TOML file into these models before `textify` turns
selected RDF nodes into embedding-ready text records. The configuration answers
four questions:

1. Which RDF types should become root embedding documents?
2. Which predicates should be used to derive labels for roots and walked
   objects?
3. How far and how broadly should outgoing graph edges be walked?
4. Which noisy or structural predicates should be skipped?

Materialized output records include a display label, embedding text, and the
source IRIs that produced that text. The first embedding-text line is always
`label: ...`. Root nodes are textified from selected outgoing predicate/object
pairs. Walked object IRIs are represented by their derived labels rather than
being expanded into full nested mini-documents.

Minimal TOML shape:

```toml
[defaults]
predicate_limit = 3
expansion_limit = 1
label_predicates = [
  "http://purl.org/dc/terms/title",
  "http://schema.org/name",
  "http://www.w3.org/2000/01/rdf-schema#label",
]
ignore_predicates = [
  "http://www.w3.org/1999/02/22-rdf-syntax-ns#first",
  "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest",
]

[label_profiles.project]
type = "https://example.org/Project"
template = "{id}: {name}"

[label_profiles.project.fields]
id = "http://purl.org/dc/terms/identifier"
name = "http://schema.org/name"

[targets.project]
type = "https://example.org/Project"
label_profile = "project"
```
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class GraphConfiguration(BaseModel):
    """Shared graph-walking and generic label configuration.

    Values in `[defaults]` are merged into each target. List values are appended
    without duplicates. Scalar values are inherited unless a target explicitly
    sets its own value.
    """

    # Ordered predicates used by `best_label` before falling back to an IRI
    # fragment. These should be graph-appropriate name/title/description
    # predicates discovered with `frink-indexing sample-types`.
    label_predicates: list[str] = Field(default_factory=list)

    # Predicates to exclude from graph walking and embedding text. Use this for
    # RDF list structure, very high-fanout edges, measurement/value layers,
    # geometry blobs, or other low-semantic-value relationships.
    ignore_predicates: list[str] = Field(default_factory=list)

    # Maximum number of objects to include per predicate for each subject.
    # When a predicate has more values than this, a stable hash-based selection
    # is used so repeated runs are deterministic.
    predicate_limit: int | None = None

    # Maximum outgoing-object depth to walk from a root. `0` includes only
    # direct root predicate/object lines. `1` allows one-hop object traversal,
    # although walked objects are represented by labels in embedding text.
    expansion_limit: int = 1

    # Whether `rdfs:label` should be included as a label predicate even when it
    # is not listed explicitly in `label_predicates`.
    include_rdfs_label: bool = True


class LabelProfileConfiguration(BaseModel):
    """Reusable label template for one RDF type.

    Profiles are optional. They are useful when a class has weak direct labels
    or when helper classes need good labels while being walked, without making
    those helper classes root textification targets.
    """

    # RDF type IRI this profile applies to.
    type: str

    # Python-style replacement template. Placeholders are aliases from
    # `fields`, for example `{id}: {name}`.
    template: str

    # Mapping from template placeholder aliases to direct predicate IRIs. Field
    # values are read from the node itself; object values are label-resolved.
    fields: dict[str, str] = Field(default_factory=dict)


class TargetConfiguration(GraphConfiguration):
    """Configuration for one root RDF type to materialize as documents."""

    # RDF type IRI whose subjects should become root embedding documents.
    type: str

    # Name of a profile in `[label_profiles]` to use for root labels. If unset,
    # the textifier uses `label_template`, then `label_predicates`, then an IRI
    # fallback.
    label_profile: str | None = None

    # Target-local template for root labels. Prefer `label_profile` when the
    # same labeling rule should also apply to walked helper nodes.
    label_template: str | None = None

    # Target-local placeholder-to-predicate map used by `label_template`.
    label_fields: dict[str, str] = Field(default_factory=dict)


class ResolvedTargetConfiguration(TargetConfiguration):
    """A target after defaults have been merged in."""

    pass


class MaterializationConfiguration(BaseModel):
    """Top-level TOML configuration for `frink-indexing textify`."""

    # Global defaults merged into every target.
    defaults: GraphConfiguration = Field(default_factory=GraphConfiguration)

    # Optional named label profiles. A target can reference a profile by name,
    # and walked object nodes can use any profile matching their RDF type.
    label_profiles: dict[str, LabelProfileConfiguration] = Field(
        default_factory=dict
    )

    # Named root targets to materialize. Each key is a local config name, not an
    # RDF identifier.
    targets: dict[str, TargetConfiguration]

    @staticmethod
    def from_toml(path: Path):
        with path.open("rb") as fd:
            data = tomllib.load(fd)
        return MaterializationConfiguration.model_validate(data)

    def iter_targets(self):
        for target in self.targets:
            yield self.for_target(target)

    def for_target(self, name: str):
        target = self.targets[name]
        defaults = self.defaults

        label_predicates = [*defaults.label_predicates]
        for predicate in target.label_predicates:
            if predicate not in label_predicates:
                label_predicates.append(predicate)

        ignore_predicates = [*defaults.ignore_predicates]
        for predicate in target.ignore_predicates:
            if predicate not in ignore_predicates:
                ignore_predicates.append(predicate)

        predicate_limit = None
        if "predicate_limit" in defaults.model_fields_set:
            predicate_limit = defaults.predicate_limit
        if "predicate_limit" in target.model_fields_set:
            predicate_limit = target.predicate_limit

        expansion_limit = defaults.expansion_limit
        if "expansion_limit" in target.model_fields_set:
            expansion_limit = target.expansion_limit

        include_rdfs_label = True
        if "include_rdfs_label" in defaults.model_fields_set:
            include_rdfs_label = defaults.include_rdfs_label
        if "include_rdfs_label" in target.model_fields_set:
            include_rdfs_label = target.include_rdfs_label

        return ResolvedTargetConfiguration(
            type=target.type,
            label_predicates=label_predicates,
            ignore_predicates=ignore_predicates,
            predicate_limit=predicate_limit,
            expansion_limit=expansion_limit,
            include_rdfs_label=include_rdfs_label,
            label_profile=target.label_profile,
            label_template=target.label_template,
            label_fields=target.label_fields,
        )

    def label_profile_for_type(
        self,
        type_iri: str,
    ) -> LabelProfileConfiguration | None:
        for profile in self.label_profiles.values():
            if profile.type == type_iri:
                return profile
        return None
