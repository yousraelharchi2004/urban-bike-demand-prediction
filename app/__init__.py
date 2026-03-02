from flask import Flask


def create_app():
    """Application factory function."""
    app = Flask(__name__)

    # Load config from environment
    import os
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

    # Register the legacy monolithic blueprint (routes.py)
    # This ensures full backward compatibility with all 40+ existing endpoints.
    from app.routes import main
    app.register_blueprint(main)

    # Register new modular blueprints.
    # These provide cleaner, reorganized versions of key endpoints.
    # Any overlapping routes with the legacy blueprint should be avoided;
    # the new blueprints use url_prefix or unique endpoint names.
    from app.blueprints.pages import pages
    from app.blueprints.prediction import prediction
    from app.blueprints.analytics import analytics
    from app.blueprints.export import export
    from app.blueprints.features import features

    app.register_blueprint(pages, url_prefix='/v2')
    app.register_blueprint(prediction, url_prefix='/v2')
    app.register_blueprint(analytics, url_prefix='/v2')
    app.register_blueprint(export, url_prefix='/v2')
    app.register_blueprint(features, url_prefix='/v2')

    return app
