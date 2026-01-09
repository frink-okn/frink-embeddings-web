from dataclasses import dataclass
from pathlib import Path
from typing import cast

from flask import Flask, current_app
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


@dataclass
class AppContext:
    client: QdrantClient
    collection: str
    model: SentenceTransformer
    graph_catalog: Path

    @property
    def graphs(self) -> list[str]:
        try:
            return sorted(self.graph_catalog.read_text().splitlines())
        except Exception:
            return []


class WrappedFlask(Flask):
    ctx: AppContext


def get_ctx() -> AppContext:
    app = cast(WrappedFlask, current_app)
    return app.ctx
