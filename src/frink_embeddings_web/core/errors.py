import httpx
from qdrant_client.http.exceptions import ResponseHandlingException


class URINotFoundError(ValueError):
    pass


def unwrap_qdrant_error(e: Exception) -> Exception:
    """Unwrap the underlying error from a qdrant-client wrapper, if present."""
    is_qdrant_wrapped = (
        isinstance(e, ResponseHandlingException)
        and e.args
        and isinstance(e.args[0], Exception)
    )

    return e.args[0] if is_qdrant_wrapped else e


def friendly_error(e: Exception) -> str:
    """A user-facing message for an exception from a Qdrant operation.

    Shared by both CLIs so connection failures read the same everywhere.
    """
    inner = unwrap_qdrant_error(e)
    if isinstance(inner, httpx.ConnectError):
        return "Could not connect to Qdrant server"
    return str(inner)
