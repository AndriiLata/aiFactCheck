import os
from functools import cached_property

class Settings:
    """Centralised runtime configuration (reads environment variables)."""

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DBPEDIA_ENDPOINT: str = os.getenv(
        "DBPEDIA_ENDPOINT", "https://dbpedia.org/sparql"
    )
    JSON_SORT_KEYS = False  # keep response order

    @cached_property
    def openai_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.OPENAI_API_KEY}"}