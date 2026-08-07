from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# This file is src/okn_embeddings/config/settings.py, so the repo root
# (where .env lives) is four levels up. That only resolves for a source
# checkout; an installed copy has no repo root and falls back to the
# defaults below.
PROJECT_ROOT = Path(__file__).parents[3]
LOCAL_ENV = PROJECT_ROOT / ".env"


class AppSettings(BaseSettings):
    """Runtime configuration, overridable by environment or .env file.

    Defaults live here rather than in a checked-in env file so that an
    installed copy of the package works without one. Precedence is
    environment variables, then .env, then these.
    """

    model_config = SettingsConfigDict(env_file=LOCAL_ENV)

    qdrant_location: str = "http://127.0.0.1:6663"
    qdrant_hnsw_ef: int = 500
    qdrant_collection: str = "OKN-Graph"
    qdrant_timeout: int = 30
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Embedding throughput knobs, both unset by default because onnxruntime's
    # own defaults win on an ordinary machine. Measured on 8 cores with
    # all-MiniLM-L6-v2: leaving both alone gives 41.6 records/s, while
    # embed_threads=1 with embed_parallel=0 gives 19.4 -- the model is small
    # enough that per-process sessions contend for the same cores rather than
    # adding throughput.
    #
    # They exist because that balance can invert: on a machine with many more
    # cores, intra-op threading stops scaling before the cores run out, and
    # splitting into processes (embed_parallel=N with embed_threads=1, so the
    # workers do not oversubscribe) can win. Measure before setting either.
    #
    # embed_threads: onnxruntime threads per session. None = its default.
    # embed_parallel: worker processes. None = no multiprocessing, 0 = one per
    # core, N = N workers.
    embed_threads: int | None = None
    embed_parallel: int | None = None


def load_settings():
    return AppSettings()
