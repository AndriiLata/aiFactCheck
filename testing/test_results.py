import pandas as pd
import pickle
import pipeline_testing
import utils
import numpy as np

# Script with various functionality for manual testing

# Small testing file to open previous results

"""""
with open("Datasets/fever_01.07_100", "rb") as f:
    data = pickle.load(f)
    df = data["results"]
    metrics = data["metrics"]

#print(df.iloc[0, 0])
print(df.iloc[1])
print(metrics)
results_df=df
"""

#Print some sample claims
"""
data = utils.load_factkg_dataset("Datasets/factkg_train.pickle")
#samples = utils.get_claims_by_indices(data, pipeline_testing.pick_test_instances(len(data), 2))
"""

#Prints factkg results by claim type
"""
def build_metadata_lookup(data):
    metadata_lookup = {}
    for claim_text, content in data.items():
        metadata_lookup[claim_text] = {
            "types": content.get("types", []),
        }
    return metadata_lookup

def extract_metadata_for_claim(claim_text, metadata_lookup):
    entry = metadata_lookup.get(claim_text, {})
    types = entry.get("types", [])

    claim_style = next((t for t in types if t.startswith("coll:") or t == "written"), None)
    reasoning_type = next((t for t in types if t in ['num1', 'num2', 'num4', 'existence', 'multi claim', 'multi hop', 'negation']), None)
    substitution = 'substitution' in types

    return pd.Series([claim_style, reasoning_type, substitution])

def compute_accuracy(df, groupby_field):
    grouped = df.groupby(groupby_field)
    result = []
    for group, group_df in grouped:
        total = len(group_df)
        correct = (group_df['true_label'] == group_df['predicted_label']).sum()
        accuracy = correct / total * 100 if total > 0 else np.nan
        result.append((group, total, correct, round(accuracy, 2)))
    return pd.DataFrame(result, columns=[groupby_field, "Total", "Correct", "Accuracy (%)"])

metadata_lookup = build_metadata_lookup(data)

results_df[['claim_style', 'reasoning_type', 'substitution']] = results_df['claim'].apply(
    lambda c: extract_metadata_for_claim(c, metadata_lookup)
)

accuracy_by_style = compute_accuracy(results_df, 'claim_style')
print("\nAccuracy by Claim Style:\n", accuracy_by_style)

# 4. Compute accuracy by reasoning type
accuracy_by_reasoning = compute_accuracy(results_df, 'reasoning_type')
print("\nAccuracy by Reasoning Type:\n", accuracy_by_reasoning)

# 5. Optional: Compute accuracy by substitution used / not used
accuracy_by_substitution = compute_accuracy(results_df, 'substitution')
print("\nAccuracy by Substitution Use:\n", accuracy_by_substitution)

"""


#Pretty print of FactKG samples
"""
def print_factkg_samples(samples):
    for idx, (claim, label, entities, types) in enumerate(samples, 1):
        print(f"\n--- Sample {idx} ---")
        print(f"Claim:        {claim}")
        print(f"Label:        {label}")
        print(f"Claim Style:  {next((t for t in types if t.startswith('coll:') or t == 'written'), 'N/A')}")
        print(
            f"Reasoning:    {next((t for t in types if t.startswith('num') or t in ['existence', 'multi claim', 'multi hop', 'negation']), 'N/A')}")
        print(f"Substitution: {'substitution' in types}")

        print("\nEntities & Relations:")
        for entity, rels in entities.items():
            relation_list = [f"[{', '.join(r)}]" if isinstance(r, list) else r for r in rels]
            print(f"  - {entity}: {'; '.join(relation_list)}")

#Print some samples from dataset

data = utils.load_factkg_dataset("Datasets/factkg_train.pickle")
#data=utils.load_fever_dataset("Datasets/fever_train.jsonl")
samples = utils.get_claims_by_indices(data, pipeline_testing.pick_test_instances(len(data), 15))

print(samples)
print_factkg_samples(samples)
"""

#Prints avg time per process step and metrics
"""
with open("Datasets/fever_api_results100.pkl", "rb") as f:
    df = pickle.load(f)

results=df["results"]
timing_infos = results["timing_info"]
timing_dicts = [ti for ti in timing_infos if isinstance(ti, dict)]

all_keys = set()
for ti in timing_dicts:
    all_keys.update(ti.keys())

averages = {}
for key in sorted(all_keys):
    values = [ti.get(key) for ti in timing_dicts if key in ti]
    avg = sum(values) / len(values) if values else 0.0
    averages[key] = avg

print("Average time per step (in seconds):")
for key, value in averages.items():
    print(f"{key:30s} : {value:.3f}")

pipeline_testing.print_metrics(df["results"])
"""

# Prints all claims where we get error
"""
# Load the dataset
with open("Datasets/fever_api_resultsTest.pkl", "rb") as f:
    df = pickle.load(f)
results=df["results"]

# Filter for rows where the predicted label is "Error"
error_df = results[results["predicted_label"] == "Error"]

# Extract the claims
error_claims = error_df["claim"].tolist()

# Print all error claims
print(f"\nTotal claims with label 'Error': {len(error_claims)}\n")
for i, claim in enumerate(error_claims, start=1):
    print(f"{i:3d}. {claim}")
"""

"""
from refined.inference.processor import Refined
refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                  entity_set="wikipedia")
spans = refined.process_text("Barack Obama was born in Hawaii")
print(spans)
"""

#extracts all claims from previous runs which give NEI
import pickle
import pandas as pd
from pathlib import Path

def extract_nei_cases_from_files(file_paths: list[str]) -> pd.DataFrame:
    """
    Given a list of pickle output files (each containing a dict with "results" DataFrame),
    loads each one, extracts the rows where predicted_label == "Not Enough Info",
    and returns a single concatenated DataFrame of [claim, true_label, source_file].
    """
    all_nei = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"[!] File not found: {file_path}, skipping.")
            continue

        with path.open("rb") as f:
            data = pickle.load(f)
        df = data.get("results")
        if df is None or "predicted_label" not in df.columns:
            print(f"[!] No results DataFrame in {file_path}, skipping.")
            continue

        nei_df = df[df["predicted_label"] == "Not Enough Info"][["claim", "true_label"]]
        nei_df = nei_df.copy()
        nei_df["source_file"] = path.name
        all_nei.append(nei_df)

    if not all_nei:
        print("No NEI cases found in any file.")
        return pd.DataFrame(columns=["claim", "true_label", "source_file"])

    combined = pd.concat(all_nei, ignore_index=True)
    return combined

file_list = [
    "Datasets/fever_01.07_100",
    "Datasets/fever_01.07_1002",
    "Datasets/fever_01.07_1003",
    "Datasets/fever_01.07_LessNEI",
    # add as many as you like...
]

nei_cases = extract_nei_cases_from_files(file_list)
print(f"Total NEI cases across all files: {len(nei_cases)}")
print(nei_cases.head())

# Save out the combined NEI cases for reuse:
nei_cases.to_pickle("Datasets/combined_NEI_cases.pkl")
nei_cases.to_csv("Datasets/combined_NEI_cases.csv", index=False)