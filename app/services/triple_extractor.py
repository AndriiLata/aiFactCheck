"""
Hybrid triple extraction:
1. Try Stanford-OpenIE locally (fast, free).
2. Fallback to GPT only if OpenIE produced < 2 triples.
"""
from __future__ import annotations

import json
from typing import Optional, List

#from .openai_client import chat
#from .models import Triple
               # starts an internal CoreNLP JVM

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large").to("cuda")
gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 1,
}

# Text to extract triplets from
text = "Punta Cana is a resort town in the municipality of Higüey, in La Altagracia Province, the easternmost province of the Dominican Republic."

# Tokenizer text
model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt')

# Generate
generated_tokens = model.generate(
    model_inputs["input_ids"].to(model.device),
    attention_mask=model_inputs["attention_mask"].to(model.device),
    **gen_kwargs,
)

# Extract text
decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

# Extract triplets
for idx, sentence in enumerate(decoded_preds):
    print(f'Prediction triplets sentence {idx}')
    print(extract_triplets(sentence))

def _openie_extract(claim: str) -> List[Triple]:
    # ToDO!!!

    return []


# --------------------------------------------------------------------------- #
# 1) GPT fallback
# --------------------------------------------------------------------------- #
_FUNC_SCHEMA = [{
    "name": "extract_triples",
    "description": (
        "Parse the input claim into ONE OR MORE subject-predicate-object triples "
        "that fully capture the factual assertions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject":   {"type": "string"},
                        "predicate": {"type": "string"},
                        "object":    {"type": "string"},
                    },
                    "required": ["subject", "predicate", "object"],
                },
            }
        },
        "required": ["triples"],
    },
}]


def _llm_extract(claim: str) -> List[Triple]:
    sys = {
        "role": "system",
        "content": (
            "You are a precise OpenIE system. Return ONLY a tool call that extracts "
            "every factual triple in the claim. Use concise noun phrases."
        ),
    }
    usr = {"role": "user", "content": claim}
    msg = chat([sys, usr], functions=_FUNC_SCHEMA)

    if not getattr(msg, "tool_calls", None):
        return []
    args = json.loads(msg.tool_calls[0].function.arguments)
    return [Triple(**t) for t in args.get("triples", [])]


# --------------------------------------------------------------------------- #
# 2) Public helper
# --------------------------------------------------------------------------- #
def parse_claim_to_triple(claim: str) -> Optional[Triple]:
    """
    • Run OpenIE first.
    • If OpenIE produced < 2 triples, call GPT.
    • Return the FIRST triple, else None.
    """
    triples = _openie_extract(claim)
    if len(triples) < 2:                         # 0 or 1 → use GPT fallback
        triples = _llm_extract(claim)
    return triples[0] if triples else None
