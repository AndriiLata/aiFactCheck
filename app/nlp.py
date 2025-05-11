"""
Light-weight NLP helper built on spaCy.

• Keeps the loaded pipeline in module scope (singleton).
• Filters entities to the types most useful for factual claims.
"""
from functools import lru_cache
import spacy

_ALLOWED_TYPES = {
    # canonical “who / what / where”
    "PERSON", "ORG", "GPE", "LOC", "NORP",
    # measurements, dates, etc.
    "DATE", "TIME", "QUANTITY", "PERCENT", "MONEY",
    "CARDINAL", "ORDINAL",
    # events, works, products, laws…
    "EVENT", "WORK_OF_ART", "LAW", "PRODUCT", "LANGUAGE",
}

@lru_cache  # ensures the model is only loaded once
def _nlp():
    return spacy.load("en_core_web_sm")

def extract_entities(sentence: str) -> list[str]:
    doc = _nlp()(sentence)
    ents = {ent.text.strip() for ent in doc.ents if ent.label_ in _ALLOWED_TYPES}
    return sorted(ents)