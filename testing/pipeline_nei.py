import random
from testing import utils
import requests
from tqdm import tqdm
import pandas as pd
import os
from app.infrastructure.llm.llm_client import chat
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Script started")  # At the very top
API_URL = "http://127.0.0.1:5000/api/verify_crewAI"
MAX_SAMPLES = 5  # Set your desired maximum here

# 1. Load and filter NEI samples
data = utils.load_fever_dataset("testing/Datasets/fever_train.jsonl", drop_NEI=False)
nei_samples = []
for claim, entry in data.items():
    label = entry["Label"][0]  # Label is a list, take first element
    evidence = entry["Evidence"]
    
    if label == "NOT ENOUGH INFO":
        nei_samples.append((claim, label, evidence))

print(f"Loaded {len(nei_samples)} NEI samples")

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
            print(f"API Error {response.status_code} for claim: {claim[:50]}...")
            pred_label = "Error"
            found_evidence = []
            reason = f"HTTP {response.status_code}"
    except Exception as e:
        print(f"Exception for claim: {claim[:50]}... -> {e}")
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

print(f"Number of results: {len(results)}")

# Print some sample results to debug
print("\nSample results:")
for i, result in enumerate(results[:3]):
    print(f"Result {i+1}:")
    print(f"  True: {result['true_label']}")
    print(f"  Predicted: {result['predicted_label']}")
    print(f"  Claim: {result['claim'][:80]}...")
    print()

# 3. LLM explanation for misclassified NEI
def ask_llm_about_nei(claim, evidence):
    sys_prompt = (
        "You are analyzing fact-checking results. "
        "Does the evidence clearly support or refute the claim, or is it insufficient? "
        "If insufficient, explain why. If sufficient, explain why."
    )
    user_prompt = (
        f"Claim: {claim}\n"
        f"Evidence found: {evidence if evidence else 'No evidence found'}"
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        msg = chat(messages)
        return msg.content
    except Exception as e:
        return f"LLM Error: {str(e)}"


# Add LLM explanations for misclassified cases
print("Generating LLM explanations for misclassified cases...")
for entry in tqdm(results, desc="LLM explanations"):
    # Check if the prediction is NOT "Not Enough Info" 
    normalized_pred = entry["predicted_label"].lower().replace(" ", "").replace("_", "")
    if normalized_pred not in ["notenoughinfo", "notinfo"]:
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
df.drop(columns=['reason']).to_csv(filename, index=False, sep=';')
print(f"Results saved to {filename}")

# Print value counts to debug
print("\nValue counts for predictions:")
print(df["predicted_label"].value_counts())
print("\nValue counts for true labels:")
print(df["true_label"].value_counts())

# Print classification report with proper label handling
print("\nClassification Report:")
try:
    # Get unique labels from both true and predicted
    all_labels = sorted(list(set(df["true_label"].tolist() + df["predicted_label"].tolist())))
    print("Unique labels found:", all_labels)
    
    report = classification_report(
        df["true_label"], 
        df["predicted_label"],
        labels=all_labels,
        zero_division=0
    )
    print(report)
except Exception as e:
    print(f"Error in classification report: {e}")

# Print confusion matrix
print("\nConfusion Matrix:")
try:
    cm = confusion_matrix(df["true_label"], df["predicted_label"])
    print(cm)
except Exception as e:
    print(f"Error in confusion matrix: {e}")

# Print accuracy
print(f"\nAccuracy: {accuracy_score(df['true_label'], df['predicted_label']):.3f}")

print(f"\nFirst few rows of results:")
print(df[["claim", "true_label", "predicted_label"]].head())