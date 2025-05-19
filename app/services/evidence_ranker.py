"""
FactGenius-style path scorer.
– Uses both edge-predicate similarities
– Penalises longer paths
– Adds literal-match bonus if middle node text appears verbatim in the claim
"""
from __future__ import annotations

import re, math
from typing import List, Tuple

from rapidfuzz import fuzz

from .models import Triple, Edge


def _last(fragment: str) -> str:
    return fragment.split("/")[-1].split("#")[-1]


def _is_literal(node: str) -> bool:
    return not node.startswith(("http://", "https://"))


class EvidenceRanker:
    def __init__(self, claim_text: str) -> None:
        self.claim_lower = claim_text.lower()

    # ---------------------------- scoring helpers ---------------------- #
    @staticmethod
    def _sim(a: str, b: str) -> float:
        return fuzz.WRatio(a, b) / 100.0          # 0-1

    def _score_path(self, triple: Triple, path: List[Edge]) -> float:
        """
        Path length ∈ {1, 2}.  Return ∈ [0,1] (can be negative before clamp).
        """
        edge_scores = [
            self._sim(_last(e.predicate), _last(triple.predicate))
            for e in path
        ]
        score = sum(edge_scores) / len(edge_scores)      # mean sim
        score += 0.15 if self._sim(_last(path[-1].object), triple.object) > 0.7 else 0
        score -= 0.05 * (len(path) - 1)                  # hop penalty

        # literal bonus on the mid node of 2-hop
        if len(path) == 2 and _is_literal(path[0].object):
            mid = path[0].object.lower()
            if mid and mid in self.claim_lower:
                score += 0.10

        return max(score, 0.0)                           # clamp below 0

    # ---------------------------- public API --------------------------- #
    def top_k(
        self,
        triple: Triple,
        paths: List[List[Edge]],
        *,
        k: int = 3,
    ) -> List[List[Edge]]:
        scored = [(self._score_path(triple, p), p) for p in paths]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:k]]

    # Backwards-compat single best
    def best_evidence(
        self,
        triple: Triple,
        paths: List[List[Edge]],
    ) -> Tuple[List[Edge], float]:
        scored = self.top_k(triple, paths, k=1)
        if scored:
            best_path = scored[0]
            return best_path, self._score_path(triple, best_path)
        return [], 0.0
