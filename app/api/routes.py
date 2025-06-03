"""
REST endpoint /verify implementing:
• 1-hop fetch & rank
• only fetch 2-hop if score < 0.4
"""
from __future__ import annotations

from flask import Blueprint, request, jsonify
from http import HTTPStatus

from app.services.triple_extractor import parse_claim_to_triple
from app.services.kg_client        import KGClient
from app.services.evidence_ranker  import EvidenceRanker
from app.services.verifier         import Verifier
from app.api import api_bp

from app.services.entity_linker import EntityLinker


@api_bp.route("/verify", methods=["POST"])
def verify():  # → (dict, int)
    data  = request.get_json(force=True)
    claim = data.get("claim")
    if not claim:
        return jsonify({"error": "JSON body must contain 'claim'"}), HTTPStatus.BAD_REQUEST

    # 1) Triple extraction
    triple = parse_claim_to_triple(claim)
    if triple is None:
        return jsonify({"error": "Could not extract a semantic triple from the claim."}), HTTPStatus.UNPROCESSABLE_ENTITY

    # 2) Entity linking -> parse the subject and object into a proper DBpedia link
    linker = EntityLinker()
    s_uris = [u for u, _ in linker.link(triple.subject)]
    o_uris = [u for u, _ in linker.link(triple.object)]

    #3) Fetch paths from DBpedia
    kg = KGClient()
    paths = kg.fetch_paths(s_uris, o_uris)

    # LLM fallback if no evidence found
    verifier = Verifier()
    if not paths:
        print("fallback: use LLM to classify the claim")
        label, reason = verifier.llm_fallback_classify(claim, triple)
        return jsonify({
            "triple":   triple.__dict__,
            "evidence": [],
            "label":    label,
            "reason":   reason,
        }), HTTPStatus.OK
    
    # 3) Rank the path, considering the claim
    ranker = EvidenceRanker(claim, triple)
    evidence_paths = ranker.top_k(triple, paths, k=3)
    (best_path, score) = evidence_paths[0] if evidence_paths else []
    


    # 4) Verification step (GPT)
    label, reason = verifier.classify(claim, triple, best_path, score)

    return jsonify({
        "triple":   triple.__dict__,
        "evidence": [e.__dict__ for e in best_path],
        "label":    label,
        "reason":   reason,
    }), HTTPStatus.OK
