# app/services/evidence_ranker.py
from __future__ import annotations
import re # Not strictly used in the provided snippet but often useful with text
import math # Same as re
import spacy
from typing import List, Tuple
# from rapidfuzz import fuzz # Not used in the provided snippet
from .models import Triple, Edge
from sentence_transformers import SentenceTransformer, util


def _last(fragment: str) -> str:
    """Extracts the last part of a URI."""
    return fragment.split("/")[-1].split("#")[-1]


class EvidenceRanker:
    # Load models once when the class is defined (shared among instances)
    try:
        nlp = spacy.load("en_core_web_md")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except OSError:
        print("Downloading Spacy en_core_web_md model and/or SentenceTransformer all-MiniLM-L6-v2 model...")
        print("Please ensure you have run: python -m spacy download en_core_web_md")
        # SentenceTransformer downloads automatically if not present, but Spacy needs explicit download.
        # Raising an error or providing a fallback might be necessary for production.
        # For now, assuming models will be available.
        raise

    def __init__(self, claim_text: str, triple: Triple) -> None:
        """
        Initializes the EvidenceRanker with the claim and the extracted triple.
        Pre-processes claim and triple elements for efficient scoring.
        """
        self.claim_text = claim_text
        self.original_triple = triple # Store original triple if needed for other context

        # Pre-process claim for sentence similarity
        self.claim_emb = EvidenceRanker.model.encode(self.claim_text, convert_to_tensor=True)
        
        # Pre-process triple's subject and object for word/entity similarity
        self.subj_doc = EvidenceRanker.nlp(triple.subject)
        self.obj_doc  = EvidenceRanker.nlp(triple.object)
    
    def _path_to_text(self, path: List[Edge]) -> str:
        """Converts an evidence path (list of edges) to a single string."""
        if not path:
            return ""
        # Uses the subject of the first edge and then predicate-object pairs for all edges
        return " ".join(
            [_last(path[0].subject)] +
            [f"{_last(e.predicate)} {_last(e.object)}" for e in path]
        )

    def _score_path(self, path: List[Edge]) -> float:
        """
        Scores a single evidence path against the claim and triple stored in the instance.
        Note: This method re-encodes the path text for sentence similarity. 
        For multiple paths, top_k is more efficient due to batch encoding.
        """
        if not path:
            return 0.0

        # 1. Penalize paths containing owl#sameAs as a direct predicate in the first or last hop
        #    (assuming these are less direct evidence, this rule can be refined)
        if path[0].predicate.endswith("owl#sameAs") or \
           path[-1].predicate.endswith("owl#sameAs"):
            return 0.0  # Low score for such paths

        # 2. Sentence similarity between claim and path text
        path_text = self._path_to_text(path)
        if not path_text: # Should not happen if path is not empty, but as a safeguard
            return 0.0
        path_emb = EvidenceRanker.model.encode(path_text, convert_to_tensor=True)
        sim_sentence = float(util.pytorch_cos_sim(self.claim_emb, path_emb)[0])

        # 3. Word/Entity similarity for subject and object
        # Compare the original triple's subject/object (pre-processed) with the
        # start/end entities of the evidence path.
        path_start_node_text = _last(path[0].subject)
        path_end_node_text = _last(path[-1].object)

        subj_sim = self.subj_doc.similarity(EvidenceRanker.nlp(path_start_node_text))
        obj_sim  = self.obj_doc.similarity(EvidenceRanker.nlp(path_end_node_text))
        
        # 4. Combine scores (weights can be tuned)
        # Current weights: 50% sentence similarity, 25% subject similarity, 25% object similarity
        score = sim_sentence * 0.5 + subj_sim * 0.25 + obj_sim * 0.25
        
        return max(score, 0.0) # Ensure score is not negative

    def top_k(
        self,
        paths: List[List[Edge]], # Takes a list of paths (each path is List[Edge])
        *,
        k: int = 3,
    ) -> List[Tuple[List[Edge], float]]: # Returns list of (path, score) tuples
        """
        Ranks a list of evidence paths and returns the top k paths with their scores.
        Uses batch processing for efficiency where possible.
        """
        if not paths:
            return []

        # 1. Filter out paths with owl#sameAs as a predicate in the first or last edge.
        #    This is a heuristic; owl#sameAs can be useful for entity alignment but
        #    might indicate a less direct evidence path if it's the main predicate.
        filtered_paths: List[List[Edge]] = []
        for path in paths:
            if not path:  # Skip empty paths
                continue
            # Check the predicate of the first and last edge in the path
            if path[0].predicate.endswith("owl#sameAs") or \
               path[-1].predicate.endswith("owl#sameAs"):
                continue # Skip this path
            filtered_paths.append(path)
        
        if not filtered_paths:
            return []

        # 2. Generate path texts for sentence similarity
        path_texts = [self._path_to_text(p) for p in filtered_paths]

        # 3. Batch encode path texts for sentence similarity with the claim
        #    This is more efficient than encoding one by one.
        path_embs = EvidenceRanker.model.encode(path_texts, convert_to_tensor=True, batch_size=32)
        sim_sentences_tensor = util.pytorch_cos_sim(self.claim_emb, path_embs)[0]  # Shape: [num_filtered_paths]

        scored_paths: List[Tuple[float, List[Edge]]] = [] # Store (score, path) for sorting

        for i, path in enumerate(filtered_paths):
            # Sentence similarity component (already batched)
            sim_sentence = float(sim_sentences_tensor[i])

            # Word/Entity similarity for subject and object
            # Compare original triple's S/O with the evidence path's S/O
            path_start_node_text = _last(path[0].subject)
            path_end_node_text = _last(path[-1].object)

            subj_sim = self.subj_doc.similarity(EvidenceRanker.nlp(path_start_node_text))
            obj_sim  = self.obj_doc.similarity(EvidenceRanker.nlp(path_end_node_text))
            
            # Combine scores (same logic as _score_path)
            current_score = sim_sentence * 0.5 + subj_sim * 0.25 + obj_sim * 0.25
            current_score = max(current_score, 0.0) # Ensure score is not negative
            
            scored_paths.append((current_score, path))

        # Sort paths by score in descending order
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k paths with their scores, ensuring the format is (path, score)
        return [(path, score) for score, path in scored_paths[:k]]

    # Backwards-compatibility single best (if you still have code using this)
    # Note: `routes.py` now handles getting the best path and score from `top_k`'s result.
    def best_evidence(
        self,
        paths: List[List[Edge]],
    ) -> Tuple[List[Edge], float]:
        """Returns the single best evidence path and its score."""
        top_results = self.top_k(paths, k=1)
        if top_results:
            return top_results[0]  # Returns (path, score)
        return [], 0.0