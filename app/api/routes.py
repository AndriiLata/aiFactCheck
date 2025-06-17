from __future__ import annotations

from http import HTTPStatus
from typing import List, Tuple

from flask import request, jsonify

import time

from . import api_bp
from ..core.extraction.triple_extractor import parse_claim_to_triple
from ..core.linking.entity_linker import EntityLinker
from ..infrastructure.kg.kg_client import KGClient
from ..core.ranking.evidence_ranker import EvidenceRanker
from ..core.verification.verifier import Verifier
from ..models import Triple, Edge, EntityCandidate

from app.config import Settings

settings = Settings()


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

    dbp = time_step("Entity_linking_subject", linker.link, claim)
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
        label, reason = verifier.classify(claim, [], 0.0)
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

    best_path, score = ranked[0]
    all_top = [p for p, _ in ranked]

    # 5 ── verification -----------------------------------------------------
    label, reason = time_step("6. final_verification", verifier.classify, claim, best_path, score)

    return jsonify(
        {
            "claim": claim,
            "evidence": [e.__dict__ for e in best_path],
            "all_top_evidence_paths": [[e.__dict__ for e in p] for p in all_top],
            "label": label,
            "reason": reason,
            "entity_linking": {
                "candidates": dbp,
            },
            **({"timing_info": timing_info} if settings.TIME_STEPS else {}),
        }
    ), HTTPStatus.OK
