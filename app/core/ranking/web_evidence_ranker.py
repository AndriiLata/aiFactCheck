"""
Web evidence ranker using the existing EvidenceRanker2 cross-encoder.
"""
from typing import List, Dict
from evidence_ranker2 import EvidenceRanker2

class WebEvidenceRanker:
    """Adapter to use EvidenceRanker2 for web evidence snippets."""
    
    def __init__(self, claim: str):
        # Reuse the existing ranker infrastructure
        self.ranker = EvidenceRanker2(claim_text=claim)
        self.claim = claim
    
    def rank_evidence(self, evidence: List[Dict], top_k: int = 20) -> List[Dict]:
        """Rank web evidence using the cross-encoder."""
        if not evidence:
            return []
        
        try:
            # Create claim-snippet pairs for cross-encoder
            pairs = [(self.claim, ev["snippet"]) for ev in evidence]
            
            # Use the cross-encoder directly
            cross_scores = EvidenceRanker2._cross_encoder.predict(pairs)
            
            # Add cross-encoder scores to evidence
            for ev, score in zip(evidence, cross_scores):
                ev["cross_encoder_score"] = float(score)
                
                # Combine with existing scores (if any)
                existing_score = ev.get("enhanced_final_score", ev.get("final_score", 0.5))
                ev["final_combined_score"] = 0.7 * float(score) + 0.3 * existing_score
            
            # Sort by cross-encoder score (primary) and return top-k
            ranked = sorted(evidence, key=lambda x: x["cross_encoder_score"], reverse=True)
            return ranked[:top_k]
            
        except Exception as e:
            print(f"Cross-encoder ranking failed: {e}")
            # Fallback to existing ranking
            return sorted(evidence, key=lambda x: x.get("enhanced_final_score", 0.5), reverse=True)[:top_k]