"""
Domain-to-trust mapping.
Falls back to hard-coded defaults if the YAML is missing or malformed.

"""
from __future__ import annotations

import os
import yaml
import tldextract

_DEFAULTS = {
    # KGs
    "wikidata.org": 0.1,
    "dbpedia.org": 0.1,
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

_YAML_PATH = os.getenv("TRUST_YAML", os.path.join(os.path.dirname(__file__), "trust_scores.yml"))


def _load_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[trust] Could not read YAML ({exc}); using defaults.")
    return {}


_DOMAIN_PRIORS = {**_DEFAULTS, **_load_yaml(_YAML_PATH)}


def score_for_url(url: str) -> float:
    """
    Return trust score ∈ [0,1] for a URL or KG name.
    """
    if url in ("wikidata", "dbpedia"):
        return 0.5

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
