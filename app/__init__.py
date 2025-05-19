from flask import Flask
from dotenv import load_dotenv
from .config import Settings
from .api import api_bp

def create_app() -> Flask:
    """Application factory."""
    load_dotenv()
    app = Flask(__name__)

    # set in Settings class (see below)
    app.config.from_object(Settings())

    # register blueprints
    app.register_blueprint(api_bp)

    return app