from __future__ import annotations

from http import HTTPStatus


from flask import request, jsonify

from . import api_bp
from ..core.crew.pipeline import verify_claim_crew

from app.config import Settings
settings = Settings()


@api_bp.route("/verify_crewAI", methods=["POST"])
def verify_crewAI():
    data = request.get_json(force=True)
    claim = data.get("claim")
    mode = data.get("mode", "hybrid")  # Default to hybrid
    mode="hybrid"
    use_cross_encoder = data.get("use_cross_encoder", True)  # Default to cross-encoder
    
    if not claim:
        return jsonify({"error": "JSON body must contain 'claim'"}), HTTPStatus.BAD_REQUEST
    
    if mode not in ["hybrid", "web_only", "kg_only"]:
        return jsonify({"error": "Mode must be 'hybrid', 'web_only', or 'kg_only'"}), HTTPStatus.BAD_REQUEST

    print(f"Running verification in {mode} mode")
    print(f"Using {'cross-encoder' if use_cross_encoder else 'bi-encoder'} for evidence ranking")
    
    out = verify_claim_crew(claim, mode=mode, use_cross_encoder=use_cross_encoder, classifierDbpedia="LLM", classifierBackup="LLM")
    return jsonify(out), HTTPStatus.OK

