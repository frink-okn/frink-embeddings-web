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
