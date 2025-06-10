"""
Aggregate weighted NLI votes into a final verdict.
Now logs per-evidence WEIGHT = trust × confidence.
"""
from typing import List, Dict, Tuple


def _aggregate(
    evidence: List[Dict], nli_out: List[Dict]
) -> Tuple[str, float, List[Dict]]:
    support, refute = 0.0, 0.0
    final_ev = []

    for ev, nl in zip(evidence, nli_out):
        weight = ev["trust"] * nl["confidence"]

        # log weight so you can inspect noise patterns
        ev_entry = {
            "snippet": ev["snippet"],
            "source": ev["source"],
            "nli": nl["label"],
            "confidence": round(nl["confidence"], 3),
            "trust": round(ev["trust"], 2),
            "weight": round(weight, 3),
        }
        final_ev.append(ev_entry)

        if nl["label"] == "entailment":
            support += weight
        elif nl["label"] == "contradiction":
            refute += weight

    total = support + refute
    if total > 0.6 and max(support, refute) / total > 0.7:
        label = "Supported" if support > refute else "Refuted"
    else:
        label = "Not Enough Info"

    confidence = round(max(support, refute), 3)
    print(  # console log for quick debugging
        f"[verdict] support={support:.3f} refute={refute:.3f} label={label}"
    )
    return label, confidence, final_ev
