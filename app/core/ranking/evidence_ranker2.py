from __future__ import annotations
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from ...models import Triple, Edge


def _last(fragment: str) -> str:
    return fragment.split("/")[-1].split("#")[-1]


class EvidenceRanker2:
    """
    Flexible two-stage evidence ranker:
    - Optional Bi-encoder filtering
    - Cross-encoder reranking
    """

    _bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def __init__(self, *, claim_text: str) -> None:
        self._claim = claim_text
        self._claim_emb = EvidenceRanker2._bi_encoder.encode(claim_text, convert_to_tensor=True)

    @staticmethod
    def _path_to_text(path: List[Edge]) -> str:
        if not path:
            return ""

        # 1) Flatten one level of nesting (in case you passed [[Edge], Edge, …])
        flat: List[Edge] = []
        for step in path:
            if isinstance(step, list):
                flat.extend(step)
            else:
                flat.append(step)

        # 2) Build the pseudo‐sentence from subject → (pred obj)*
        parts = [_last(flat[0].subject)]
        for e in flat:
            parts.append(_last(e.predicate))
            parts.append(_last(e.object))
        return " ".join(parts)

    def top_k(
        self,
        paths: List[List[Edge]],
        *,
        k: int = 3,
        filter_k: Optional[int] = 200,
        use_bi_encoder: bool = True,
    ) -> List[Tuple[List[Edge], float]]:
        """
        Parameters:
            paths          – List of candidate KG paths
            k              – Number of final top results to return
            filter_k       – How many to keep after bi-encoder stage (None = use all)
            use_bi_encoder – Whether to apply bi-encoder filtering before reranking
        """
        if not paths:
            return []

        texts = [self._path_to_text(p) for p in paths]

        if use_bi_encoder:
            embs = EvidenceRanker2._bi_encoder.encode(texts, convert_to_tensor=True, batch_size=32)
            bi_scores = util.pytorch_cos_sim(self._claim_emb, embs)[0]

            ranked = sorted(
                zip(paths, texts, bi_scores),
                key=lambda x: float(x[2]),
                reverse=True
            )
            if filter_k is not None:
                ranked = ranked[:filter_k]

            rerank_paths = [path for path, _, _ in ranked]
            rerank_texts = [txt for _, txt, _ in ranked]

        else:
            rerank_paths = paths
            rerank_texts = texts

        rerank_pairs = [(self._claim, txt) for txt in rerank_texts]
        cross_scores = EvidenceRanker2._cross_encoder.predict(rerank_pairs)

        reranked = sorted(
            zip(rerank_paths, cross_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [(p, float(s)) for p, s in reranked[:k]]