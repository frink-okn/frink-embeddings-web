from typing import cast

from flask import Flask, current_app

from ..config import AppContext


class WrappedFlask(Flask):
    ctx: AppContext


def get_ctx() -> AppContext:
    app = cast(WrappedFlask, current_app)
    return app.ctx
