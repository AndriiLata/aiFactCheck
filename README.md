# Multi-Agent Fact-Checking Pipeline

**Multi-agent reasoning system** that verifies short natural-language claims by fusing knowledge-graph evidence with real-time web search results.

---
## 1. Results

...

## 2. How to Access with the Hosted API

| Hosted URL | HTTPS `POST` endpoint |
|----------|-----------------------|
| `https://test.api` | `/api/verify` |

### 2.1 Request body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `claim` | `string` | **Yes** | – | The statement you want verified. |
| `mode` | `"hybrid"` &#124; `"web_only"` &#124; `"kg_only"` | No | `"hybrid"` | • **hybrid**: try KG first, fall back to web. <br>• **web_only**: skip KG, use web search only. <br>• **kg_only**: KG evidence only, no web search. |
| `use_cross_encoder` | `bool` | No | `true` | Ranking method for web snippets:<br>• `true` – slower but more accurate MiniLM cross-encoder.<br>• `false` – faster bi-encoder (all-MiniLM-L6‐v2). |
| `classifierDbpedia` | `"LLM"` &#124; `"DEBERTA"` | No | `"LLM"` | Algorithm for *KG* evidence: GPT-4 (`LLM`) vs. DeBERTa-v3 (`DEBERTA`). |
| `classifierBackup` | `"LLM"` &#124; `"DEBERTA"` | No | `"LLM"` | Algorithm for *web* evidence when KG is inconclusive. |

### 2.2 Example request

```bash
curl -X POST https://test.api/api/verify \
  -H "Content-Type: application/json" \
  -d '{
        "claim": "Mount Kilimanjaro is the highest mountain in Africa",
        "mode": "hybrid",
        "use_cross_encoder": true,
        "classifierDbpedia": "LLM",
        "classifierBackup": "DEBERTA"
      }'
```

### 2.3 Response schema

```bash
{
  "claim": "...",
  "label": "Supported | Refuted | Not Enough Info",
  "confidence": 0.83,          // present only for DEBERTA paths
  "reason": "...",             // GPT-style one-liner
  "evidence": [...],           // snippets or KG paths
  "mode": "hybrid, LLM, DEBERTA",
  "evidence_count": 5,
  "ranking_method": "cross_encoder",
  "kg_success": true
}
```

## 3. How to Run Locally
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download SpaCy model
python -m spacy download en_core_web_md

# 3. Add credentials
echo "OPENAI_API_KEY=sk-...
AZURE_API_KEY=<optional>
AZURE_ENDPOINT=<optional>" > .env

# 4. Launch the API
python run.py
# → server starts on http://127.0.0.1:8000  (Flask defaults)

```

