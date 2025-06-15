from __future__ import annotations

from typing import List, Tuple

import spacy
from sentence_transformers import SentenceTransformer, util

from ...models import Triple, Edge
import torch

def _last(fragment: str) -> str:
    return fragment.split("/")[-1].split("#")[-1]


class EvidenceRanker:
    """
    Scores KG paths against the claim.
    """

    _nlp = spacy.load("en_core_web_md")
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(self, *, claim_text: str, triple: Triple) -> None:
        self.device = "cpu"
        #"cuda" if torch.cuda.is_available() else
        self._claim_emb = EvidenceRanker._model.encode(claim_text, convert_to_tensor=True, device=self.device)
        print("Using device:", self.device)
        self._subj_doc = EvidenceRanker._nlp(triple.subject)
        self._obj_doc = EvidenceRanker._nlp(triple.object)
        

    # ------------------------------------------------------------------ #
    @staticmethod
    def _path_to_text(path: List[Edge]) -> str:
        if not path:
            return ""
        return " ".join(
            [_last(path[0].subject)]
            + [f"{_last(e.predicate)} {_last(e.object)}" for e in path]
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def is_weak_predicate(pred: str) -> bool:
        return any(keyword in pred for keyword in [
            "wikiPage", "sameAs", "seeAlso", "externalLink", "redirects",
            "label", "comment", "subject", "page", "type", "url"
        ])

    # ------------------------------------------------------------------ #
    def top_k(self, paths: List[List[Edge]], *, k: int = 3) -> List[Tuple[List[Edge], float]]:
        if not paths:
            return []

        # Filter out paths
        filtered = [
            p for p in paths
                if not (self.is_weak_predicate(p[0].predicate) or self.is_weak_predicate(p[-1].predicate))
        ]       
        if not filtered:
            return []

        # SentenceTransformer encoding and similarity
        texts = [self._path_to_text(p) for p in filtered]
        embs = EvidenceRanker._model.encode(texts, convert_to_tensor=True, batch_size=32, device=self.device)
        sent_sims = util.pytorch_cos_sim(self._claim_emb, embs)[0]

        # Pre-rank top 100 by Sentence
        scored_sent_only = [(float(sent_sims[idx]), path) for idx, path in enumerate(filtered)]
        scored_sent_only.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_sent_only[:100]

        # Apply SpaCy similarity only on top 100
        scored: List[Tuple[float, List[Edge]]] = []
        for sim_score, path in top_candidates:
            start = _last(path[0].subject)
            end = _last(path[-1].object)
            subj_sim = self._subj_doc.similarity(EvidenceRanker._nlp(start))
            obj_sim = self._obj_doc.similarity(EvidenceRanker._nlp(end))
            final_score = sim_score * 0.7 + subj_sim * 0.25 + obj_sim * 0.25
            scored.append((final_score, path))
            #print(f"SBERT: {sim_score:.4f}, Subj: {subj_sim:.4f}, Obj: {obj_sim:.4f}")

        # Return top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(p, s) for s, p in scored[:k]]
