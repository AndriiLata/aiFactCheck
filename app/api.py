from flask import Blueprint, request, jsonify

from .nlp import extract_linked_entities
from .dbpedia import fetch_hundred_triples   # unchanged

bp = Blueprint("api", __name__)

@bp.route("/triples", methods=["POST"])
def triples():
    """
    Body: { "sentence": "Obama is a president of USA" }

    Response:
    {
      "entities": ["Obama", "USA"],
      "triples": [
          { "subject": "http://dbpedia.org/resource/Barack_Obama",
            "predicate": "http://dbpedia.org/ontology/office",
            "object": "http://dbpedia.org/resource/President_of_the_United_States"
          },
          ...
      ]
    }
    """
    payload = request.get_json(silent=True)
    if not payload or "sentence" not in payload:
        return jsonify({"error": "POST body must be JSON with a 'sentence' field"}), 400

    sentence: str = payload["sentence"]

    # [{text, label, uri, score}, …]
    linked = extract_linked_entities(sentence)

    # keep old “entities” list for backward compatibility
    entity_texts = [e["text"] for e in linked]

    all_triples: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for ent in linked:
        # use canonical DBpedia URI (works because of the dbpedia.py patch)
        for t in fetch_hundred_triples(ent["uri"]):
            key = (t["subject"], t["predicate"], t["object"])
            if key not in seen:
                seen.add(key)
                all_triples.append(t)

    return jsonify({"entities": entity_texts, "triples": all_triples})
