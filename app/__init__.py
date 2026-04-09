import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from flask import Flask, redirect, request


def _register_modular_blueprints(app: Flask, url_prefix: str = "", name_prefix: str = ""):
    """Register the modular route stack once for a given URL prefix."""
    from app.blueprints.analytics import analytics
    from app.blueprints.export import export
    from app.blueprints.features import features
    from app.blueprints.pages import pages
    from app.blueprints.prediction import prediction

    app.register_blueprint(pages, url_prefix=url_prefix, name_prefix=name_prefix)
    app.register_blueprint(prediction, url_prefix=url_prefix, name_prefix=name_prefix)
    app.register_blueprint(analytics, url_prefix=url_prefix, name_prefix=name_prefix)
    app.register_blueprint(export, url_prefix=url_prefix, name_prefix=name_prefix)
    app.register_blueprint(features, url_prefix=url_prefix, name_prefix=name_prefix)


def create_app():
    """Application factory function."""
    secret = os.environ.get("SECRET_KEY")
    if not secret:
        raise RuntimeError("SECRET_KEY must be set in the environment before starting the app")
    app = Flask(__name__)
    app.config["SECRET_KEY"] = secret

    _register_modular_blueprints(app, url_prefix="/v2", name_prefix="v2")

    @app.before_request
    def redirect_to_v2():
        if request.path.startswith("/v2") or request.path.startswith("/static") or request.path.startswith("/_debugger") or request.path == "/favicon.ico":
            return
        return redirect(f"/v2{request.path}", code=307)

    return app
