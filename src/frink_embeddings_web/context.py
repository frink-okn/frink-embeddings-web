from dataclasses import dataclass
from typing import cast

from flask import Flask, current_app
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from frink_embeddings_web.settings import AppSettings


@dataclass
class AppContext:
    client: QdrantClient
    model: SentenceTransformer
    settings: AppSettings

    @staticmethod
    def from_settings(settings: AppSettings) -> "AppContext":
        client = QdrantClient(
            location=settings.qdrant_location,
            timeout=settings.qdrant_timeout,
        )
        model = SentenceTransformer(settings.model_name)

        return AppContext(
            client=client,
            model=model,
            settings=settings,
        )

    @property
    def graphs(self) -> list[str]:
        try:
            return sorted(self.settings.graph_catalog.read_text().splitlines())
        except FileNotFoundError:
            return []


class WrappedFlask(Flask):
    ctx: AppContext


def get_ctx() -> AppContext:
    app = cast(WrappedFlask, current_app)
    return app.ctx
