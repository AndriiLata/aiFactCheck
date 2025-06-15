"""
Domain-to-trust mapping.
Falls back to hard-coded defaults if the YAML is missing or malformed.

"""
from __future__ import annotations


import tldextract

_DEFAULTS = {
    # KGs
    "wikidata.org": 1.0,
    "dbpedia.org": 1.0,
    # high quality news / refs
    ".gov": 0.95,
    ".edu": 0.95,
    "nytimes.com": 0.9,
    "bbc.co.uk": 0.9,
    # social
    "reddit.com": 0.1,
    # fallback
    "*": 0.5,
}



_DOMAIN_PRIORS = {**_DEFAULTS}


def score_for_url(url: str) -> float:
    """
    Return trust score ∈ [0,1] for a URL or KG name.
    """
    if url in ("wikidata", "dbpedia"):
        return 1.0

    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    # exact
    if domain in _DOMAIN_PRIORS:
        return _DOMAIN_PRIORS[domain]

    # suffix match like ".gov"
    for suf, sc in _DOMAIN_PRIORS.items():
        if suf.startswith(".") and domain.endswith(suf):
            return sc

    return _DOMAIN_PRIORS.get("*", 0.5)
