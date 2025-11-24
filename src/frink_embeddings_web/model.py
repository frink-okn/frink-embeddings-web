from typing import Annotated, Literal

from pydantic import BaseModel, Field


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
    graphs: list[str] | None
    limit: int = 10
    offset: int = 0
