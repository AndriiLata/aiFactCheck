from __future__ import annotations

from http import HTTPStatus
from typing import List, Tuple

from flask import request, jsonify

from . import api_bp
from ..core.extraction.triple_extractor import parse_claim_to_triple
from ..core.linking.entity_linker import EntityLinker
from ..infrastructure.kg.kg_client import KGClient
from ..core.ranking.evidence_ranker import EvidenceRanker
from ..core.verification.verifier import Verifier
from ..models import Triple, Edge, EntityCandidate
import concurrent.futures


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
    


    # 2 ── entity linking ---------------------------------------------------
    linker = EntityLinker()
    subject_candidates = []
    object_candidates = []

    for triple in extracted:  # extracted is now a List[Triple]
        s_cands = linker.link(triple.subject, top_k=3)
        o_cands = linker.link(triple.object, top_k=3)
        subject_candidates.append([c.__dict__ for c in s_cands])
        object_candidates.append([c.__dict__ for c in o_cands])

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

    verifier = Verifier()

    # --- PARALLEL EXECUTION STARTS HERE ---
    def kg_task(triple):
        kg = KGClient()
        # Entity linking for this triple
        s_cands = linker.link(triple.subject, top_k=3)
        o_cands = linker.link(triple.object, top_k=3)
        s_dbp = [c.dbpedia_uri for c in s_cands if c.dbpedia_uri]
        s_wd = [c.wikidata_uri for c in s_cands if c.wikidata_uri]
        o_dbp = [c.dbpedia_uri for c in o_cands if c.dbpedia_uri]
        o_wd = [c.wikidata_uri for c in o_cands if c.wikidata_uri]
        paths = kg.fetch_paths(s_dbp, s_wd, o_dbp, o_wd, max_hops=2)
        if paths:
            ranker = EvidenceRanker(claim_text=claim, triple=triple)
            ranked = ranker.top_k(paths, k=5)
            best_path, score = ranked[0]
            label, reason = verifier.classify(claim, triple, best_path, score)
            return {
                "label": label,
                "reason": reason,
                "evidence": [e.__dict__ for e in best_path],
                "source": "kg"
            }
        return {
            "label": "Not Enough Info",
            "reason": "No KG evidence.",
            "evidence": [],
            "source": "kg"
        }
    
    def llm_task(triple):
        label, reason, evidence_list, uncertainty = verifier.llm_fallback_classify(claim, triple)
        return {"label": label, "reason": reason, "evidence": evidence_list, "uncertainty": uncertainty, "source": "llm"}

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for triple in extracted:
            futures.append(executor.submit(kg_task, triple))
            futures.append(executor.submit(llm_task, triple))
        results = [f.result() for f in futures]

    # --- EVALUATION STEP ---
    labels = [r["label"].lower() for r in results]

    # Count occurrences
    from collections import Counter
    label_counts = Counter(labels)
    num_false = label_counts.get("refuted", 0) + label_counts.get("false", 0)
    num_true = label_counts.get("supported", 0) + label_counts.get("true", 0)
    num_not_enough = label_counts.get("not enough info", 0)

    if num_false > 0:
        final_label = "Refuted"
    elif num_true == len(labels):
        final_label = "Supported"
    elif num_not_enough == len(labels):
        final_label = "Not Enough Info"
    else:
        final_label = label_counts.most_common(1)[0][0].capitalize()

    # Combine reasons/evidence for transparency
    combined_reason = " | ".join([f"{r['source']}: {r['reason']}" for r in results])
    combined_evidence = []
    for r in results:
        combined_evidence.extend(r["evidence"])

    return jsonify(
    {
        "claim": claim,
        "triples": [t.__dict__ for t in extracted],
        "label": final_label,
        "reason": combined_reason,
        "evidence": combined_evidence,
        "num_false": num_false,
        "num_true": num_true,
        "num_not_enough_info": num_not_enough,
    }
), HTTPStatus.OK


'''
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
  '''  



