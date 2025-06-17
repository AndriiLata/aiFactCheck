from __future__ import annotations
import os
from functools import cached_property
from typing import Literal


class Settings:
    """
    Centralised runtime configuration pulled from environment variables.
    """

    # ---- LLM -----------------------------------------------------------
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    AZURE_API_KEY: str = os.getenv("AZURE_API_KEY", "")
    AZURE_ENDPOINT: str = os.getenv("AZURE_ENDPOINT", "")

    PROVIDER_IN_USE: Literal["openai", "azure"] = os.getenv(
        "PROVIDER_IN_USE", "openai"
    ).lower()

    # ---- Knowledge graphs ---------------------------------------------
    DBPEDIA_ENDPOINT: str = os.getenv(
        "DBPEDIA_ENDPOINT", "https://dbpedia.org/sparql"
    )
    DBPEDIA_ENDPOINT_PUBLIC: str = os.getenv(
        "DBPEDIA_ENDPOINT_PUBLIC", "https://dbpedia.org/sparql"
    )

    # -------------------------------------------------------------------
    JSON_SORT_KEYS = False  # keep original order in Flask jsonify
    TIME_STEPS = True

    # convenience
    @cached_property
    def openai_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.OPENAI_API_KEY}"}
