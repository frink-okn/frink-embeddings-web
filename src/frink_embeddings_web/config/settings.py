from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# This file is src/frink_embeddings_web/config/settings.py, so the repo root
# (where default.env / .env live) is four levels up.
PROJECT_ROOT = Path(__file__).parents[3]
DEFAULT_ENV = PROJECT_ROOT / "default.env"
LOCAL_ENV = PROJECT_ROOT / ".env"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(DEFAULT_ENV, LOCAL_ENV))
    qdrant_location: str
    qdrant_hnsw_ef: int
    qdrant_collection: str
    qdrant_timeout: int
    model_name: str


def load_settings():
    return AppSettings()  # type: ignore
