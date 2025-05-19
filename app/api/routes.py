from flask import request, jsonify
from app.api import api_bp

from app.services.entity_extractor import extract_linked_entities
from app.services.dbpedia_client    import fetch_top_similar_entities, fetch_all_triples

@api_bp.route("/triples", methods=["POST"])
def triples():
    """
    POST { "sentence": "..." }
    → {
         "entities": [
           { "text": "Donald Trump", "links": [ ... up to 10 {uri,score} ... ] },
           ...
         ],
         "triples": [
           {"subject": "...", "predicate": "...", "object": "..."},
           ...
         ]
       }
    """
    data = request.get_json(force=True)
    sentence = data.get("sentence")
    if not sentence:
        return jsonify({"error": "JSON must include 'sentence'"}), 400

    # 1) Extract entities (LLM + spaCy + numeric)
    ents = extract_linked_entities(sentence)

    # 2) Build per-entity link lists
    entities_out = []
    for ent in ents:
        txt = ent["text"]
        if txt.isdigit():
            links = []
        else:
            links = fetch_top_similar_entities(txt, top_n=10)
        entities_out.append({"links": links, "text": txt})

    # 3) Fetch & dedupe triples from all URIs
    triples = []
    seen = set()
    for ent in entities_out:
        for link in ent["links"]:
            uri = link["uri"]
            for row in fetch_all_triples(uri, batch_size=250):
                subj = row.get("subject")   or row.get("s")
                pred = row.get("predicate") or row.get("p")
                obj  = row.get("object")    or row.get("o")
                key = (subj, pred, obj)
                if key not in seen:
                    seen.add(key)
                    triples.append({"subject": subj, "predicate": pred, "object": obj})

    return jsonify({"entities": entities_out, "triples": triples})
