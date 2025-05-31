import pickle


# Class containing the utils specific to the FactKG Dataset


# Loads the FactKG dataset from a pickle file
def loadfactkg_dataset(path: str) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


# Returns a claim and label at a given index
def get_claim_entry_by_index(dataset: dict, index: int):
    # Returns the (claim, label, entry) tuple at the given index in the dataset
    keys = list(dataset.keys())
    if index < 0 or index >= len(keys):
        raise IndexError("Index out of range.")
    claim = keys[index]
    entry = dataset[claim]
    label_orig = entry.get("Label")[0]
    label = normalize_label(label_orig)
    return claim, label


# Returns a list of claims and labels specified by th indices in the input
def get_claims_by_indices(dataset: dict, indices: list[int]):
    results = []
    for index in indices:
        result = get_claim_entry_by_index(dataset, index)
        results.append(result)
    return results


# Transforms the factKG labels to our internal format
def normalize_label(label: str) -> str:
    # Normalize ground-truth labels to standard format
    label = str(label).lower()
    if label == "true":
        return "Supported"
    elif label == "false":
        return "Refuted"
    return "Not Enough Info"
