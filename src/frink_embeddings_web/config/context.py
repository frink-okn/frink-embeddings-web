from dataclasses import dataclass

from qdrant_client import QdrantClient

from ..core.embedding import Embedder, make_embedder
from ..core.graphs import get_graphs
from .settings import AppSettings, load_settings


@dataclass
class AppContext:
    client: QdrantClient
    embedder: Embedder
    settings: AppSettings

    @staticmethod
    def from_env() -> "AppContext":
        return AppContext.from_settings(load_settings())

    @staticmethod
    def from_settings(settings: AppSettings) -> "AppContext":
        client = QdrantClient(
            location=settings.qdrant_location,
            timeout=settings.qdrant_timeout,
        )
        embedder = make_embedder(settings)

        return AppContext(
            client=client,
            embedder=embedder,
            settings=settings,
        )

    @property
    def graphs(self) -> list[str]:
        return get_graphs(self)
