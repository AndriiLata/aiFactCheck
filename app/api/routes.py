from __future__ import annotations

from http import HTTPStatus
from typing import List, Tuple
import time


from flask import request, jsonify

from . import api_bp
from ..core.extraction.triple_extractor import parse_claim_to_triple
from ..core.linking.entity_linker import EntityLinker
from ..infrastructure.kg.kg_client import KGClient
from ..core.ranking.evidence_ranker import EvidenceRanker
from ..core.verification.verifier import Verifier
from ..core.verification.web_verifier import WebVerifier
from ..models import Triple, Edge, EntityCandidate
from ..core.crew.pipeline import verify_claim_crew
from ..core.crew.pipeline_paraphrase import verify_claim_paraphrased

from app.config import Settings
settings = Settings()

@api_bp.route("/verify_web_only", methods=["POST"])
def verify_web_only():
    data = request.get_json(force=True)
    claim = data.get("claim")
    if not claim:
        return jsonify({"error": "JSON body must contain 'claim'"}), HTTPStatus.BAD_REQUEST

    out = verify_claim_paraphrased(claim)
    return jsonify(out), HTTPStatus.OK


@api_bp.route("/verify_crewAI", methods=["POST"])
def verify_crewAI():
    data = request.get_json(force=True)
    claim = data.get("claim")
    if not claim:
        return jsonify({"error": "JSON body must contain 'claim'"}), HTTPStatus.BAD_REQUEST

    out = verify_claim_crew(claim)
    return jsonify(out), HTTPStatus.OK


@api_bp.route("/verify_rag", methods=["POST"])
def verify_rag():
    data = request.get_json(force=True)
    claim = data.get("claim")
    if not claim:
        return jsonify({"error": "JSON body must contain 'claim'"}), HTTPStatus.BAD_REQUEST

    # 1 ── triple extraction ------------------------------------------------
    extracted: Triple | None = parse_claim_to_triple(claim)
    if extracted is None:
        return jsonify(
            {
                "claim": claim,
                "triple": None,
                "evidence": [],
                "label": "Not Enough Info",
                "reason": "Could not extract a semantic triple from the claim.",
                "entity_linking": None,
            }
        ), HTTPStatus.OK
    

    # --- FORCE LLM FALLBACK FOR TESTING ---
    web_verifier = WebVerifier()
    label, reason, evidence_list = web_verifier.verify(claim, extracted)
    return jsonify(
        {
            "claim": claim,
            "triple": extracted.__dict__,
            "evidence": evidence_list,
            "label": label,
            "reason": reason,
            "entity_linking": None,
        }
    ), HTTPStatus.OK


@api_bp.route("/verify", methods=["POST"])
def verify():
    data = request.get_json(force=True)
    claim = data.get("claim")
    if not claim:
        return jsonify({"error": "JSON body must contain 'claim'"}), HTTPStatus.BAD_REQUEST

    # 1 ── triple extraction ------------------------------------------------
    extracted: Triple | None = parse_claim_to_triple(claim)
    if extracted is None:
        return jsonify(
            {
                "claim": claim,
                "triple": None,
                "evidence": [],
                "label": "Not Enough Info",
                "reason": "Could not extract a semantic triple from the claim.",
                "entity_linking": None,
            }
        ), HTTPStatus.OK

    # 2 ── entity linking ---------------------------------------------------
    linker = EntityLinker()
    s_cands: List[EntityCandidate] = linker.link(extracted.subject, top_k=3)
    o_cands: List[EntityCandidate] = linker.link(extracted.object, top_k=3)

    s_dbp = [c.dbpedia_uri for c in s_cands if c.dbpedia_uri]
    s_wd = [c.wikidata_uri for c in s_cands if c.wikidata_uri]
    o_dbp = [c.dbpedia_uri for c in o_cands if c.dbpedia_uri]
    o_wd = [c.wikidata_uri for c in o_cands if c.wikidata_uri]

    if not (s_dbp or s_wd) or not (o_dbp or o_wd):
        return jsonify(
            {
                "claim": claim,
                "triple": extracted.__dict__,
                "evidence": [],
                "label": "Not Enough Info",
                "reason": "Could not link entities to KG URIs.",
                "entity_linking": {
                    "subject_candidates": [c.__dict__ for c in s_cands],
                    "object_candidates": [c.__dict__ for c in o_cands],
                },
            }
        ), HTTPStatus.OK

    # 3 ── KG paths ---------------------------------------------------------
    kg = KGClient()
    paths: List[List[Edge]] = kg.fetch_paths(s_dbp, s_wd, o_dbp, o_wd, max_hops=2)

    verifier = Verifier()

    # 4 ── rank evidence ----------------------------------------------------
    if paths:
        ranker = EvidenceRanker(claim_text=claim, triple=extracted)
        ranked: List[Tuple[List[Edge], float]] = ranker.top_k(paths, k=5)
        best_path, score = ranked[0]
        all_top = [p for p, _ in ranked]
        label, reason = verifier.classify(claim, extracted, best_path, score)
    else:
        best_path = []
        all_top = []
        label, reason = verifier.classify(claim, extracted, [], 0.0)

    # 5 ── LLM fallback if still not enough info ----------------------------
    if label == "Not Enough Info":
        label, reason, evidence_list = verifier.llm_fallback_classify(claim, extracted)
        if evidence_list:
            best_path = []  # No KG evidence, but we have web evidence
            all_top = []
            # Optionally, you can add evidence_list to the output as "evidence"
            return jsonify(
                {
                    "claim": claim,
                    "triple": extracted.__dict__,
                    "evidence": evidence_list,
                    "all_top_evidence_paths": [],
                    "label": label,
                    "reason": reason,
                    "entity_linking": {
                        "subject_candidates": [c.__dict__ for c in s_cands],
                        "object_candidates": [c.__dict__ for c in o_cands],
                    },
                }
            ), HTTPStatus.OK
    
    return jsonify(
        {
            "claim": claim,
            "triple": extracted.__dict__,
            "evidence": [e.__dict__ for e in best_path],
            "all_top_evidence_paths": [[e.__dict__ for e in p] for p in all_top],
            "label": label,
            "reason": reason,
            "entity_linking": {
                "subject_candidates": [c.__dict__ for c in s_cands],
                "object_candidates": [c.__dict__ for c in o_cands],
            },
        }
    ), HTTPStatus.OK

@api_bp.route("/verify2", methods=["POST"])
def verify2():
    data = request.get_json(force=True)
    claim = data.get("claim")
    if not claim:
        return jsonify({"error": "JSON body must contain 'claim'"}), HTTPStatus.BAD_REQUEST

    # timing function
    timing_info = {}

    def time_step(label, func, *args, **kwargs):
        if not settings.TIME_STEPS:
            return func(*args, **kwargs)

        start = time.time()
        result = func(*args, **kwargs)
        difference = time.time() - start
        timing_info[label] = difference
        # Update total time after every step
        previous_total = timing_info.get("0. total_time", 0.0)
        timing_info["0. total_time"] = previous_total + difference
        return result

    # 2 ── entity linking ---------------------------------------------------
    from ..core.linking.entity_linker2 import EntityLinker2
    linker = EntityLinker2()

    dbp = time_step("1. Entity_linking", linker.link, claim)
    print("entities ", dbp)

    if not dbp:
        return jsonify(
            {
                "claim": claim,
                "evidence": [],
                "label": "Not Enough Info",
                "reason": "Could not link entities to KG URIs.",
                "entity_linking": {
                    "candidates": dbp,
                },
                **({"timing_info": timing_info} if settings.TIME_STEPS else {}),
            }
        ), HTTPStatus.OK

    # 3 ── KG paths ---------------------------------------------------------
    from ..infrastructure.kg.kg_client2 import KGClient2
    kg = KGClient2()
    paths: List[List[Edge]] = time_step("4. kg_path_retrieval", kg.fetch_paths, dbp)

    from ..core.verification.verifier2 import Verifier2

    verifier = Verifier2()

    if not paths:
        label, reason = verifier.classify(claim, [])
        return jsonify(
            {
                "claim": claim,
                "evidence": [],
                "label": label,
                "reason": reason,
                "entity_linking": {
                    "candidates": dbp,
                },
                **({"timing_info": timing_info} if settings.TIME_STEPS else {}),
            }
        ), HTTPStatus.OK

    # 4 ── rank evidence ----------------------------------------------------
    from ..core.ranking.evidence_ranker2 import EvidenceRanker2
    ranker = EvidenceRanker2(claim_text=claim)
    ranked: List[Tuple[List[Edge], float]] = time_step("5. evidence_ranking", ranker.top_k, paths, k=10,
                                                       use_bi_encoder=False)

    def pretty_print_ranked_paths(ranked_paths: List[Tuple[List[Edge], float]]) -> None:
        """
        Print each ranked path on its own line, showing the human‐readable triples
        and the score. Works for 1- or 2-edge paths.
        """

        def _short(uri: str) -> str:
            return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]

        print(f"\nTop {len(ranked_paths)} ranked paths:")
        for i, (path, score) in enumerate(ranked_paths, 1):
            # flatten subject → predicate → (mid → predicate →) object
            elems = []
            for edge in path:
                elems.append(_short(edge.subject))
                elems.append(_short(edge.predicate))
            elems.append(_short(path[-1].object))

            line = " → ".join(elems)
            print(f"[{i:2d}] {line}  (score={score:.4f})")

    print("Claim:", claim)
    pretty_print_ranked_paths(ranked)

    all_top = [p for p, _ in ranked]

    # 5 ── verification -----------------------------------------------------
    evidence=[edge for path, _ in ranked for edge in path]
    label, reason = time_step("6. final_verification", verifier.classify, claim, evidence)

    return jsonify(
        {
            "claim": claim,
            "all_top_evidence_paths": [[e.__dict__ for e in p] for p in all_top],
            "label": label,
            "reason": reason,
            "entity_linking": {
                "candidates": dbp,
            },
            **({"timing_info": timing_info} if settings.TIME_STEPS else {}),
        }
    ), HTTPStatus.OK

