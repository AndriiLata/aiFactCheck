"""
FactGenius-style path scorer.
– Uses both edge-predicate similarities
– Penalises longer paths
– Adds literal-match bonus if middle node text appears verbatim in the claim
"""
from __future__ import annotations
import re, math
import spacy
from typing import List, Tuple
from rapidfuzz import fuzz
from .models import Triple, Edge
from sentence_transformers import SentenceTransformer, util


def _last(fragment: str) -> str:
    return fragment.split("/")[-1].split("#")[-1]


class EvidenceRanker:
    nlp = spacy.load("en_core_web_md")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def __init__(self, claim_text: str) -> None:
        self.claim = claim_text

    # ---------------------------- scoring helpers ---------------------- #
    @staticmethod
    def _sim_Sentence(a: str, b: str) -> float:
        emb = EvidenceRanker.model.encode([a, b], convert_to_tensor=True)
        return float(util.pytorch_cos_sim(emb[0], emb[1]))
    
    @staticmethod
    def _sim_Word(a: str, b: str) -> float:
        doc_a = EvidenceRanker.nlp(a)
        doc_b = EvidenceRanker.nlp(b)
        return doc_a.similarity(doc_b)  # returns float in [0,1]
    
    def _path_to_text(self, path: List[Edge]) -> str:
        return " ".join(
        [ _last(path[0].subject) ] +
        [ f"{_last(e.predicate)} {_last(e.object)}" for e in path ])    # returns float in [0,1]


    def _score_path(self, triple: Triple, path: List[Edge]) -> float:
        edge_score = self._sim_Sentence(self.claim, self._path_to_text(path)) * 0.5 + self._sim_Word(triple.subject, path[0].subject) * 0.25 + self._sim_Word(triple.object, path[-1].object) * 0.25
        if path[0].predicate.__contains__("owl#sameAs") or path[-1].predicate.__contains__("owl#sameAs"):
           return 0.0
        #   print(edge_score)
        return max(edge_score, 0.0)                           # clamp below 0

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
