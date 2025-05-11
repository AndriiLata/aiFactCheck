from flask import Blueprint, request, jsonify

from .nlp import extract_entities
from .dbpedia import fetch_all_triples, fetch_hundred_triples

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
    entities = extract_entities(sentence)

    all_triples: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for ent in entities:
        for t in fetch_hundred_triples(ent):
            key = (t["subject"], t["predicate"], t["object"])
            if key not in seen:
                seen.add(key)
                all_triples.append(t)

    return jsonify({"entities": entities, "triples": all_triples})
