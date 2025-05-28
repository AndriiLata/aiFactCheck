import pickle
import random


def load_factkg_dataset(path: str) -> dict:
    # Loads the FactKG dataset from a pickle file
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_claim_entry_by_index(dataset: dict, index: int):
    # Returns the (claim, label, entry) tuple at the given index in the dataset
    keys = list(dataset.keys())
    if index < 0 or index >= len(keys):
        raise IndexError("Index out of range.")
    claim = keys[index]
    entry = dataset[claim]
    label_orig = entry.get("Label")[0]
    label=normalize_label(label_orig)
    return claim, label


def get_claims_by_indices(dataset: dict, indices: list[int]):
    results = []
    for index in indices:
        result = get_claim_entry_by_index(dataset, index)
        results.append(result)
    return results


def normalize_label(label: str) -> str:
    # Normalize ground-truth labels to standard format
    label = str(label).lower()
    if label == "true":
        return "Supported"
    elif label == "false":
        return "Refuted"
    return "Not Enough Info"

#data=load_factkg_dataset("Datasets/factkg_train.pickle")
#print(get_claim_entry_by_index(data, 3))

import pandas as pd
df = pd.read_pickle("Datasets/factkg_api_results.pkl")
print(df.iloc[0])
print(df.iloc[1])
print(df.iloc[2])