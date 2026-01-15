from loguru import logger

from frink_embeddings_web.context import AppContext, WrappedFlask
from frink_embeddings_web.settings import load_settings


def create_app() -> WrappedFlask:
    app = WrappedFlask(__name__)

    settings = load_settings()

    logger.info("Detected settings: " + str(settings))

    ctx = AppContext.from_settings(settings)
    app.ctx = ctx

    # Register API & Web routes
    from frink_embeddings_web.routes import api, web

    app.register_blueprint(api)
    app.register_blueprint(web)

    return app


app = create_app()


if __name__ == "__main__":
    app.run()
