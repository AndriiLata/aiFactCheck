from testing import utils
import requests
from tqdm import tqdm
import pandas as pd
from app.infrastructure.llm.llm_client import chat

API_URL = "http://127.0.0.1:5000/api/verify2"

# 1. Load and filter NEI samples
data = utils.load_fever_dataset("testing/Datasets/train.jsonl", drop_NEI=False)
nei_samples = [(claim, entry["Label"][0], entry["Evidence"]) 
               for claim, entry in data.items() 
               if entry["Label"][0].lower() == "not enough info"]

# 2. Classify NEI claims
results = []
for claim, true_label, evidence in tqdm(nei_samples, desc="Classifying NEI claims"):
    try:
        response = requests.post(API_URL, json={"claim": claim}, timeout=100)
        if response.status_code == 200:
            raw_data = response.json()
            pred_label = raw_data.get("label", "NOT_ENOUGH_INFO")
            found_evidence = raw_data.get("evidence", [])
            reason = raw_data.get("reason", "")
        else:
            pred_label = "Error"
            found_evidence = []
            reason = ""
    except Exception as e:
        pred_label = "Error"
        found_evidence = []
        reason = str(e)
    results.append({
        "claim": claim,
        "true_label": true_label,
        "predicted_label": pred_label,
        "found_evidence": found_evidence,
        "reason": reason
    })

# 3. LLM explanation for misclassified NEI
def ask_llm_about_nei(claim, evidence):
    sys = (
        f"Claim: {claim}\n"
        f"Evidence found: {evidence}\n"
        "Does this evidence clearly support or refute the claim, or is it insufficient? "
        "If insufficient, explain why. If sufficient, explain why."
    )
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": claim}
    ]
    msg = chat(messages)
    return msg.content


for entry in results:
    if entry["predicted_label"].lower() != "not enough info":
        entry["llm_explanation"] = ask_llm_about_nei(entry["claim"], entry["found_evidence"])
    else:
        entry["llm_explanation"] = None

# 4. Output
df = pd.DataFrame(results)
df.to_csv("nei_test_results.csv", index=False)
print(df.head())