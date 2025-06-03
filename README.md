# 🧠 aiFactCheck

## 1. Install Requirements

```bash
pip install -r requirements.txt
```

## 2. Run the App

```bash
python run.py
```

---

## How It Works So Far

### 1. HTTP Endpoint
**File:** `app/api.py`

- Defined under the `/triples` route.
- Validates that the JSON payload includes a `sentence` field.
- Returns JSON with two keys:
  - `entities`
  - `triples`

### 2. Entity Extraction
**File:** `app/nlp.py`
**Function:** `def extract_linked_entities()`

- Uses **spaCy** with the `en_core_web_trf` transformer pipeline.
- Identifies spans labeled `PERSON`, `ORG`, `GPE`, `LOC`, `DATE`, etc.
- Filters only allowed entity types for factual claims.

### 3. KB Linking
**File:** `app/nlp.py`
**Function:** `def _spotlight()`

- **Primary:** DBpedia Spotlight REST API via `requests`.
- Sends surface text, retrieves candidate URIs with similarity scores.
- Selects the URI with the highest confidence.

### 4. Fuzzy Fallback
**File:** `app/nlp.py`
**Function:** `def _sparql_fuzzy()`

- Used when Spotlight fails or confidence is too low.
- Queries DBpedia SPARQL endpoint for labels containing the surface mention.
- Uses **RapidFuzz** to compute Levenshtein-based similarity (`WRatio`).
- Chooses the top-matching URI (score normalized to 0–1).

### 5. Triple Retrieval

**Function:** `fetch_hundred_triples(entity_uri)` in `app/dbpedia.py`

- Runs a SPARQL `SELECT ?s ?p ?o` that finds triples where the entity is subject or object.
- Returns up to 100 triples as JSON dicts.


## 🧪 Virtuoso SPARQL Endpoint Quickstart (DBpedia)

- Spin up a local Virtuoso SPARQL endpoint with Docker and preload it with a DBpedia Databus collection.

Quickstart:

```bash
cd virtuoso-sparql-endpoint
COLLECTION_URI=https://databus.dbpedia.org/dbpedia/collections/dbpedia-snapshot-2022-03 \
VIRTUOSO_ADMIN_PASSWD=YourSecretPassword \
docker-compose up
```
- After startup, your endpoint will be available at: http://localhost:8890/sparql

📌 Tips:

- Use the preview collection (/virtuoso-sparql-endpoint-quickstart-preview) for faster testing.
- Large datasets may take hours to load.
- Modify .env to configure ports, data directories, and collection URIs.
- A more detailed README.md can be found [here](https://github.com/dbpedia/virtuoso-sparql-endpoint-quickstart/blob/master/README.md)