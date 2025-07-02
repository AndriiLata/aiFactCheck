from flask import Flask
from dotenv import load_dotenv
load_dotenv()

from .config import Settings
from .api import api_bp


def create_app() -> Flask:
    """
    Minimal Flask application factory.
    """

    app = Flask(__name__)
    app.config.from_object(Settings())  # type: ignore[arg-type]

    # blueprints
    app.register_blueprint(api_bp, url_prefix="/api")

    return app
