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

    def __init__(self, claim_text: str, triple: Triple) -> None:
        self.claim = claim_text
        self.claim_emb = EvidenceRanker.model.encode(self.claim, convert_to_tensor=True)
        self.subj_doc = EvidenceRanker.nlp(triple.subject)
        self.obj_doc  = EvidenceRanker.nlp(triple.object)
    
    def _path_to_text(self, path: List[Edge]) -> str:
        return " ".join(
        [ _last(path[0].subject) ] +
        [ f"{_last(e.predicate)} {_last(e.object)}" for e in path ])  

    # ---------------------------- public API --------------------------- #
    def top_k(
    self,
    triple: Triple,
    paths: List[List[Edge]],
    *,
    k: int = 3,
) -> List[Tuple[List[Edge], float]]:
     if not paths:
        return []

    # 1. Filter out owl#sameAs early
     filtered_paths = [
         path for path in paths
         if "owl#sameAs" not in path[0].predicate and "owl#sameAs" not in path[-1].predicate
     ]
     if not filtered_paths:
        return []

    # 2. Generate path texts
     path_texts = [self._path_to_text(p) for p in filtered_paths]

    # 3. Batch encode path texts
     path_embs = EvidenceRanker.model.encode(path_texts, convert_to_tensor=True, batch_size=16)

    # 4. Compute cosine similarity to claim embedding
     sim_sentences = util.pytorch_cos_sim(self.claim_emb, path_embs)[0]  # shape: [num_paths]

    # 5. Compute word-level similarities (subject/object)
     word_sims = []
     for path in filtered_paths:
        subj_sim = self.subj_doc.similarity(EvidenceRanker.nlp(path[0].subject))
        obj_sim  = self.obj_doc.similarity(EvidenceRanker.nlp(path[-1].object))
        word_sims.append((subj_sim, obj_sim))

    # 6. Combine scores
     scored = []
     for i, path in enumerate(filtered_paths):
        sim_sentence = float(sim_sentences[i])
        subj_sim, obj_sim = word_sims[i]
        score = sim_sentence * 0.5 + subj_sim * 0.25 + obj_sim * 0.25
        scored.append((score, path))

    # 7. Sort and return top-k
     scored.sort(key=lambda x: x[0], reverse=True)
     return [(p, s) for s, p in scored[:k]]
