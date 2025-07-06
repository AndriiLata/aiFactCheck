from __future__ import annotations

from http import HTTPStatus


from flask import request, jsonify

from . import api_bp
from ..core.crew.pipeline import verify_claim_crew

from app.config import Settings
settings = Settings()


@api_bp.route("/verify", methods=["POST"])
def verify():
    data = request.get_json(force=True)
    claim = data.get("claim")
    mode = data.get("mode", "hybrid")  # Default to hybrid
    use_cross_encoder = data.get("use_cross_encoder", True)  # Default to cross-encoder
    classifier_dbpedia = data.get("classifierDbpedia", "LLM")  # Default to LLM
    classifier_backup = data.get("classifierBackup", "LLM")
    
    if not claim:
        return jsonify({"error": "JSON body must contain 'claim'"}), HTTPStatus.BAD_REQUEST
    
    if mode not in ["hybrid", "web_only", "kg_only"]:
        return jsonify({"error": "Mode must be 'hybrid', 'web_only', or 'kg_only'"}), HTTPStatus.BAD_REQUEST

    
    out = verify_claim_crew(claim, mode=mode, use_cross_encoder=use_cross_encoder,
                            classifierDbpedia=classifier_dbpedia, classifierBackup=classifier_backup)
    return jsonify(out), HTTPStatus.OK

