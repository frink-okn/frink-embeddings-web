from frink_embeddings_web.app import create_app


def main() -> None:
    app = create_app()
    app.run()
