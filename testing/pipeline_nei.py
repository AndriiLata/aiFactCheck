import random
from testing import utils
import requests
from tqdm import tqdm
import pandas as pd
import os
from app.infrastructure.llm.llm_client import chat
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Script started")  # At the very top
API_URL = "http://127.0.0.1:5000/api/verify2"
MAX_SAMPLES = 1  # Set your desired maximum here

# 1. Load and filter NEI samples
data = utils.load_fever_dataset("testing/Datasets/train.jsonl", drop_NEI=False)
nei_samples = [(claim, entry["Label"][0], entry["Evidence"]) 
               for claim, entry in data.items() 
               if entry["Label"][0].lower() == "not enough info"]
print(f"Loaded {len(nei_samples)} NEI samples")  # After filtering

# Randomly select up to MAX_SAMPLES NEI claims
if len(nei_samples) > MAX_SAMPLES:
    nei_samples = random.sample(nei_samples, MAX_SAMPLES)

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

print(f"Number of results: {len(results)}")  # Before DataFrame creation

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

base_filename = "nei_test_results.csv"
filename = base_filename
i = 1
while os.path.exists(filename):
    filename = f"nei_test_results_{i}.csv"
    i += 1

print("Saving results to CSV...")
print("Current working directory:", os.getcwd())
df.to_csv(filename, index=False, sep=';')
print(f"Results saved to {filename}")
print(df.head())

# Print classification report
print("\nClassification Report:")
print(classification_report(df["true_label"], df["predicted_label"]))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(df["true_label"], df["predicted_label"]))

# Print accuracy
print("\nAccuracy:", accuracy_score(df["true_label"], df["predicted_label"]))