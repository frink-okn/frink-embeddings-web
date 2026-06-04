from typing import TYPE_CHECKING, Protocol

import numpy as np
from fastembed import TextEmbedding

if TYPE_CHECKING:
    from ..config.settings import AppSettings


class Embedder(Protocol):
    """Turns text into a query/index vector.

    Shared seam between the query path (this branch) and the embedding/upload
    pipeline, so both produce vectors with the same model.
    """

    def embed(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    """Embedder backed by fastembed (onnxruntime, no torch)."""

    def __init__(self, model_name: str):
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, text: str) -> np.ndarray:
        vector = next(iter(self._model.embed([text])))
        return np.asarray(vector, dtype=np.float32)


def make_embedder(settings: "AppSettings") -> Embedder:
    """Construct the configured embedder.

    The single place backend selection would branch if a non-fastembed model
    is ever needed.
    """
    return FastEmbedEmbedder(settings.model_name)
