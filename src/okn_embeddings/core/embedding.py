from typing import TYPE_CHECKING, Protocol

import numpy as np
from fastembed import TextEmbedding

if TYPE_CHECKING:
    from ..config.settings import AppSettings


class Embedder(Protocol):
    """Turns text into a query/index vector.

    Shared seam between the query path and the embedding/upload pipeline, so
    both produce vectors with the same model. `embed_many` is the batch entry
    point the bulk uploader uses; `embed` is the single-text convenience.
    """

    def embed(self, text: str) -> np.ndarray: ...

    def embed_many(self, texts: list[str]) -> list[np.ndarray]: ...


class FastEmbedEmbedder:
    """Embedder backed by fastembed (onnxruntime, no torch).

    `threads` and `parallel` are the throughput knobs (see `AppSettings`);
    both default to None, which leaves onnxruntime's own defaults in place --
    the fastest configuration on an ordinary machine, because the model is
    small enough that intra-op threading already uses every core.

    GPU execution needs no setting here: fastembed's device selection
    defaults to auto, so installing the `fastembed-gpu` package (which brings
    onnxruntime-gpu) is enough to pick up a CUDA device.
    """

    def __init__(
        self,
        model_name: str,
        threads: int | None = None,
        parallel: int | None = None,
    ):
        self._model = TextEmbedding(model_name=model_name, threads=threads)
        self.threads = threads
        self.parallel = parallel

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        return [
            np.asarray(vector, dtype=np.float32)
            for vector in self._model.embed(texts, parallel=self.parallel)
        ]


def make_embedder(settings: "AppSettings") -> Embedder:
    """Construct the configured embedder.

    The single place backend selection would branch if a non-fastembed model
    is ever needed.
    """
    return FastEmbedEmbedder(
        settings.model_name,
        threads=settings.embed_threads,
        parallel=settings.embed_parallel,
    )
