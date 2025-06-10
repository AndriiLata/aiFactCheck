from __future__ import annotations

from typing import List, Tuple

import spacy
from sentence_transformers import SentenceTransformer, util

from ...models import Triple, Edge


def _last(fragment: str) -> str:
    return fragment.split("/")[-1].split("#")[-1]


class EvidenceRanker:
    """
    Scores KG paths against the claim.
    """

    _nlp = spacy.load("en_core_web_md")
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(self, *, claim_text: str, triple: Triple) -> None:
        self._claim_emb = EvidenceRanker._model.encode(claim_text, convert_to_tensor=True)
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
    def top_k(self, paths: List[List[Edge]], *, k: int = 3) -> List[Tuple[List[Edge], float]]:
        if not paths:
            return []

        filtered = [
            p
            for p in paths
            if not (p[0].predicate.endswith("owl#sameAs") or p[-1].predicate.endswith("owl#sameAs"))
        ]
        if not filtered:
            return []

        texts = [self._path_to_text(p) for p in filtered]
        embs = EvidenceRanker._model.encode(texts, convert_to_tensor=True, batch_size=32)
        sent_sims = util.pytorch_cos_sim(self._claim_emb, embs)[0]

        scored: List[Tuple[float, List[Edge]]] = []
        for idx, path in enumerate(filtered):
            start = _last(path[0].subject)
            end = _last(path[-1].object)
            subj_sim = self._subj_doc.similarity(EvidenceRanker._nlp(start))
            obj_sim = self._obj_doc.similarity(EvidenceRanker._nlp(end))
            score = float(sent_sims[idx]) * 0.5 + subj_sim * 0.25 + obj_sim * 0.25
            scored.append((score, path))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(p, s) for s, p in scored[:k]]
