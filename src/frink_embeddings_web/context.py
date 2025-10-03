from dataclasses import dataclass
from typing import cast

from flask import Flask, current_app
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


@dataclass
class AppContext:
    client: QdrantClient
    collection: str
    model: SentenceTransformer


class WrappedFlask(Flask):
    ctx: AppContext


def get_ctx() -> AppContext:
    app = cast(WrappedFlask, current_app)
    return app.ctx
