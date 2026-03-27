import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class GraphConfiguration(BaseModel):
    label_predicates: list[str] = Field(default_factory=list)
    ignore_predicates: list[str] = Field(default_factory=list)
    predicate_limit: int | None = None
    include_rdfs_label: bool = True


class TargetConfiguration(GraphConfiguration):
    type: str


class ResolvedTargetConfiguration(TargetConfiguration):
    pass


class MaterializationConfiguration(BaseModel):
    defaults: GraphConfiguration = Field(default_factory=GraphConfiguration)
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
            include_rdfs_label=include_rdfs_label,
        )
