from loguru import logger

from ..config import AppContext
from ._flask import WrappedFlask


def create_app() -> WrappedFlask:
    app = WrappedFlask(__name__)

    ctx = AppContext.from_env()
    logger.info("Detected settings: " + str(ctx.settings))
    app.ctx = ctx

    # Register API & Web routes
    from .routes import api, web

    app.register_blueprint(api)
    app.register_blueprint(web)

    return app


app = create_app()


if __name__ == "__main__":
    app.run()
