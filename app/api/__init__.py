from flask import Blueprint

api_bp = Blueprint("api", __name__)

from . import routes  # noqa: E402, F401 circular import is fine here