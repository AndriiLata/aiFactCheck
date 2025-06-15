"""
Mini util around the DeBERTa-v3 MNLI model.

Usage:
    preds = batch_nli("claim text", ["premise #1", "premise #2", ...])
Returns:
    List[dict] – [{label: 'entailment'|'contradiction'|'neutral',
                   confidence: float}, ...]
"""
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_LABELS = ("contradiction", "neutral", "entailment")

_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-large-mnli"
).eval()


@torch.inference_mode()
def batch_nli(hypothesis: str, premises: List[str]) -> List[dict]:
    # split into reasonably sized chunks so we do not OOM
    chunk, out = 8, []
    for i in range(0, len(premises), chunk):
        batch_prem = premises[i : i + chunk]
        toks = _tokenizer(
            batch_prem,
            [hypothesis] * len(batch_prem),
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        logits = _model(**{k: v.cuda() if torch.cuda.is_available() else v for k, v in toks.items()}).logits
        probs = torch.softmax(logits, dim=-1).cpu()
        for p in probs:
            idx = int(torch.argmax(p))
            out.append({"label": _LABELS[idx], "confidence": float(p[idx])})
    return out
