from dataclasses import dataclass

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ..core.graphs import get_graphs
from .settings import AppSettings, load_settings


@dataclass
class AppContext:
    client: QdrantClient
    model: SentenceTransformer
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
        model = SentenceTransformer(settings.model_name)

        return AppContext(
            client=client,
            model=model,
            settings=settings,
        )

    @property
    def graphs(self) -> list[str]:
        return get_graphs(self)
