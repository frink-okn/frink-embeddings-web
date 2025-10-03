from typing import Annotated, Literal
from pydantic import BaseModel, Field

class TextFeature(BaseModel):
    type: Literal["text"]
    value: str
    weight: float = 1

class NodeFeature(BaseModel):
    type: Literal["node"]
    value: str
    weight: float = 1

Feature = Annotated[
    TextFeature | NodeFeature,
    Field(..., discriminator="type")
]

class Query(BaseModel):
    positive: list[Feature]
    negative: list[Feature]
