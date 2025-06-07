# 🧠 aiFactCheck

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Download SpaCy model

```bash
python -m spacy download en_core_web_md
```

### 3. Create a .env file under the root directory with the following content:

```env
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
AZURE_API_KEY=YOUR_AZURE_API_KEY
AZURE_ENDPOINT=YOUR_AZURE_ENDPOINT
```

### 4. Run the App

```bash
python run.py
```

### 5. Access the App
- URL POST http://127.0.0.1:5000/api/verify

- Headers Content‑Type: application/json

- Body (raw, JSON)

```json
{
  "claim": "TUM is a university in Germany"
}
```
- Response (JSON)

```json
{
  "claim": "TUM is a university in Germany",
  "triple": {
    "subject": "TUM",
    "predicate": "is_a",
    "object": "university in Germany"
  },
  "evidence": [
    {
      "subject": "http://dbpedia.org/resource/Technical_University_of_Munich",
      "predicate": "http://purl.org/dc/terms/subject",
      "object": "http://dbpedia.org/resource/Category:Universities_and_colleges_in_Bavaria",
      "source_kg": "dbpedia"
    }
  ],
  "all_top_evidence_paths": [],
  "label": "Supported",
  "reason": "The evidence path ... confirms that TUM is a German university.",
  "entity_linking": {
    "subject_candidates": [],
    "object_candidates": []
  }
}
```
