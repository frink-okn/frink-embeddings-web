from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class TextFeature(BaseModel):
    type: Literal["text"]
    value: str


class NodeFeature(BaseModel):
    type: Literal["node"]
    value: str


Feature = Annotated[
    TextFeature | NodeFeature,
    Field(..., discriminator="type"),
]


class Query(BaseModel):
    feature: Feature
    include_graphs: list[str] | None = None
    exclude_graphs: list[str] | None = None
    limit: int = 10
    offset: int = 0

    @model_validator(mode="after")
    def validate_graph_modes(self):
        if self.include_graphs and self.exclude_graphs:
            raise ValueError("Only one of include_graphs or exclude_graphs may be set")
        return self
