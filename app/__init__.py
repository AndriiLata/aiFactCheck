from flask import Flask
from .api import bp as api_bp

def create_app() -> Flask:
    """Factory that creates and configures the Flask application."""
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False          # keep insertion order
    app.register_blueprint(api_bp, url_prefix="/")
    return app